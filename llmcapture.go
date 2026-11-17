// Package llmcapture records every LLM call made through a wrapped client as
// two JSON messages, one before the call and one after.
package llmcapture

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"reflect"
	"time"

	"github.com/google/uuid"
	"google.golang.org/genai"
)

const (
	// SubjectRequest is the default subject for the message published before
	// each call, so a call that never returns (hang, crash, OOM kill) is still
	// on record.
	SubjectRequest = "llm.req"
	// SubjectResponse is the default subject for the message published after
	// each call that returned, failed or not.
	SubjectResponse = "llm.resp"
)

var (
	// ErrNotPersisted is wrapped by both errors below.
	ErrNotPersisted = errors.New("llmcapture: not persisted")
	// ErrRequestNotPersisted: a Required request was not stored, or a call
	// with any Required message could not be captured at all, so the LLM was
	// not called and nothing was billed. Safe to retry.
	ErrRequestNotPersisted = fmt.Errorf("%w: request", ErrNotPersisted)
	// ErrResponseNotPersisted: a Required response was not stored, so the
	// answer was withheld. The call was billed; retrying pays again.
	ErrResponseNotPersisted = fmt.Errorf("%w: response", ErrNotPersisted)
)

// WritePolicy decides what a failed publish does to the call.
type WritePolicy int

const (
	// BestEffort logs a failed publish and goes on.
	BestEffort WritePolicy = iota
	// Required fails the call when its message is not persisted: for the
	// request, the LLM is not called; for the response, the answer is
	// withheld (the call is already billed).
	Required
)

// Policy is the write policy of each message of a call.
type Policy struct {
	Request  WritePolicy
	Response WritePolicy
}

type Option func(*Capturer)

// WithSubjects sets the subjects of the request and response messages.
func WithSubjects(request, response string) Option {
	return func(c *Capturer) { c.reqSubject, c.respSubject = request, response }
}

// WithBestEffortTimeout bounds each BestEffort publish (default
// DefaultBestEffortTimeout, also used when d <= 0). It is short because a slow
// publish delays the call even though its outcome cannot fail it.
func WithBestEffortTimeout(d time.Duration) Option {
	if d <= 0 {
		d = DefaultBestEffortTimeout
	}
	return func(c *Capturer) { c.bestEffortTimeout = d }
}

const (
	// DefaultBestEffortTimeout bounds each BestEffort publish; see
	// WithBestEffortTimeout.
	DefaultBestEffortTimeout = 500 * time.Millisecond
	// requiredTimeout bounds each Required publish.
	requiredTimeout = 5 * time.Second
)

// Generator is the call being captured; *genai.Models satisfies it.
type Generator interface {
	GenerateContent(ctx context.Context, model string, contents []*genai.Content, cfg *genai.GenerateContentConfig) (*genai.GenerateContentResponse, error)
}

// Publisher sends one message; msgID dedupes retries. Required policies are
// only as strong as Publish: it must return nil only once the message is
// stored (e.g. a JetStream publish that waits for the ack).
type Publisher interface {
	Publish(ctx context.Context, subject, msgID string, payload []byte) error
}

// Request is published to SubjectRequest. Contents and Config use genai's JSON
// field names, with every byte field replaced by its {sha256, size}; inline
// data becomes {mimeType, sha256, size}.
type Request struct {
	CallID    string            `json:"call_id"`
	StartedAt time.Time         `json:"started_at"`
	Model     string            `json:"model"`
	Attrs     map[string]string `json:"attrs,omitempty"`
	Contents  any               `json:"contents"`
	Config    any               `json:"config,omitempty"`
}

// Response is published to SubjectResponse, joined to its Request on CallID.
// Model and Attrs repeat the request's so a response is attributable even when
// its request was lost.
type Response struct {
	CallID     string            `json:"call_id"`
	Model      string            `json:"model"`
	Attrs      map[string]string `json:"attrs,omitempty"`
	DurationNS int64             `json:"duration_ns"`
	Response   any               `json:"response,omitempty"`
	Error      string            `json:"error,omitempty"`
}

// Wrap returns a Capturer that calls next and publishes a message before and
// after each call, to SubjectRequest and SubjectResponse unless WithSubjects
// says otherwise. Both messages are BestEffort; see WithPolicy.
func Wrap(next Generator, pub Publisher, opts ...Option) *Capturer {
	c := &Capturer{next: next, pub: pub, reqSubject: SubjectRequest, respSubject: SubjectResponse, bestEffortTimeout: DefaultBestEffortTimeout}
	for _, o := range opts {
		o(c)
	}
	return c
}

// Capturer is a Generator that records each call.
type Capturer struct {
	next              Generator
	pub               Publisher
	reqSubject        string
	respSubject       string
	policy            Policy
	attrs             map[string]string
	bestEffortTimeout time.Duration
}

// WithAttr returns a copy of c whose calls carry key=value in both their
// messages, e.g. the ids that join a call to the caller's records; c is
// unchanged.
func (c *Capturer) WithAttr(key, value string) *Capturer {
	cp := *c
	cp.attrs = maps.Clone(c.attrs)
	if cp.attrs == nil {
		cp.attrs = map[string]string{}
	}
	cp.attrs[key] = value
	return &cp
}

// WithPolicy returns a copy of c whose calls use p; c is unchanged.
func (c *Capturer) WithPolicy(p Policy) *Capturer {
	cp := *c
	cp.policy = p
	return &cp
}

// Result is one captured call.
type Result struct {
	// Response is nil when the call failed or its answer was withheld.
	Response *genai.GenerateContentResponse
	// CallID joins the call to its messages; empty when the call was not
	// captured (BestEffort and no id could be generated).
	CallID string
	// RequestPersisted and ResponsePersisted report whether Publish returned
	// nil for each message.
	RequestPersisted  bool
	ResponsePersisted bool
}

// Generate makes the call and reports how it was captured. Unlike the usual Go
// convention, Result is non-nil alongside an error whenever a call id was
// assigned, so a failed call can still be traced.
func (c *Capturer) Generate(ctx context.Context, model string, contents []*genai.Content, cfg *genai.GenerateContentConfig) (*Result, error) {
	id, err := uuid.NewV7()
	if err != nil {
		if c.policy.Request == Required || c.policy.Response == Required {
			return nil, fmt.Errorf("%w: uuid.NewV7: %w", ErrRequestNotPersisted, err)
		}
		slog.Error("llmcapture: uuid.NewV7, call not captured", "err", err)
		resp, callErr := c.next.GenerateContent(ctx, model, contents, cfg)
		return &Result{Response: resp}, callErr
	}
	r := &Result{CallID: id.String()}
	if err := c.publishRequest(ctx, r.CallID, time.Now(), model, contents, cfg); err != nil {
		if c.policy.Request == Required {
			return r, fmt.Errorf("%w: %w", ErrRequestNotPersisted, err)
		}
		slog.Error("llmcapture: request not persisted", "call_id", r.CallID, "err", err)
	} else {
		r.RequestPersisted = true
	}
	start := time.Now()
	resp, callErr := c.next.GenerateContent(ctx, model, contents, cfg)
	d := time.Since(start)
	// Detached so a canceled call is still recorded.
	if err := c.publishResponse(context.WithoutCancel(ctx), r.CallID, model, d, resp, callErr); err != nil {
		if c.policy.Response == Required {
			return r, errors.Join(callErr, fmt.Errorf("%w: %w", ErrResponseNotPersisted, err))
		}
		slog.Error("llmcapture: response not persisted", "call_id", r.CallID, "err", err)
	} else {
		r.ResponsePersisted = true
	}
	r.Response = resp
	return r, callErr
}

// GenerateContent is Generate with genai's signature, so a Capturer drops in
// for *genai.Models.
func (c *Capturer) GenerateContent(ctx context.Context, model string, contents []*genai.Content, cfg *genai.GenerateContentConfig) (*genai.GenerateContentResponse, error) {
	r, err := c.Generate(ctx, model, contents, cfg)
	if r == nil {
		return nil, err
	}
	return r.Response, err
}

func (c *Capturer) publishRequest(ctx context.Context, callID string, start time.Time, model string, contents []*genai.Content, cfg *genai.GenerateContentConfig) error {
	msg := Request{CallID: callID, StartedAt: start.UTC(), Model: model, Attrs: c.attrs}
	var err error
	if msg.Contents, err = redacted(contents); err != nil {
		return err
	}
	if cfg != nil {
		cp := *cfg
		cp.HTTPOptions = nil // transport settings, may carry auth headers
		if msg.Config, err = redacted(&cp); err != nil {
			return err
		}
	}
	return c.publish(ctx, c.policy.Request, c.reqSubject, callID+".req", msg)
}

func (c *Capturer) publishResponse(ctx context.Context, callID, model string, d time.Duration, resp *genai.GenerateContentResponse, callErr error) error {
	msg := Response{CallID: callID, Model: model, Attrs: c.attrs, DurationNS: d.Nanoseconds()}
	if callErr != nil {
		msg.Error = callErr.Error()
	}
	if resp != nil {
		r := *resp
		r.SDKHTTPResponse = nil // transport headers, not the model's answer
		var err error
		if msg.Response, err = redacted(&r); err != nil {
			return err
		}
	}
	return c.publish(ctx, c.policy.Response, c.respSubject, callID+".resp", msg)
}

func (c *Capturer) publish(ctx context.Context, p WritePolicy, subject, msgID string, msg any) error {
	payload, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("json.Marshal(%s): %w", msgID, err)
	}
	timeout := c.bestEffortTimeout
	if p == Required {
		timeout = requiredTimeout
	}
	pctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	if err := c.pub.Publish(pctx, subject, msgID, payload); err != nil {
		return fmt.Errorf("Publisher.Publish(%s, %s, %d bytes): %w", subject, msgID, len(payload), err)
	}
	return nil
}

// redacted returns v as a JSON tree with every []byte replaced by its digest.
// The bytes are swapped for sentinels on a copy before marshaling, so blobs are
// never base64-encoded and only real byte fields are matched.
func redacted(v any) (any, error) {
	r := redactor{nonce: rand.Text(), digests: map[string]digest{}, path: map[visit]bool{}}
	cp, err := r.clone(reflect.ValueOf(v))
	if err != nil {
		return nil, err
	}
	b, err := json.Marshal(cp.Interface())
	if err != nil {
		return nil, fmt.Errorf("json.Marshal(%T): %w", v, err)
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	var tree any
	if err := dec.Decode(&tree); err != nil {
		return nil, fmt.Errorf("json.Decoder.Decode(%T): %w", v, err)
	}
	return r.replace(tree), nil
}

var (
	marshalerType = reflect.TypeFor[json.Marshaler]()
	// blobTypes have their Data flattened into them: {mimeType, sha256, size}.
	blobTypes = map[reflect.Type]bool{
		reflect.TypeFor[genai.Blob]():                 true,
		reflect.TypeFor[genai.FunctionResponseBlob](): true,
	}
)

type redactor struct {
	nonce string
	// digests maps the base64 of each sentinel to the bytes it stands for.
	digests map[string]digest
	// path holds the references being cloned, to catch cycles.
	path map[visit]bool
}

type digest struct {
	fields map[string]any
	// flatten merges fields into the enclosing object, for a blob's data.
	flatten bool
}

type visit struct {
	ptr uintptr
	typ reflect.Type
	len int
}

// clone deep-copies v, swapping each non-empty []byte for a sentinel.
func (r *redactor) clone(v reflect.Value) (reflect.Value, error) {
	switch v.Kind() {
	case reflect.Pointer, reflect.Map, reflect.Slice:
		if v.IsNil() {
			return v, nil
		}
	case reflect.Interface:
		if v.IsNil() {
			return v, nil
		}
		e, err := r.clone(v.Elem())
		if err != nil {
			return v, err
		}
		i := reflect.New(v.Type()).Elem()
		i.Set(e)
		return i, nil
	case reflect.Struct:
		c := reflect.New(v.Type()).Elem()
		c.Set(v)
		for i := range v.NumField() {
			f := v.Type().Field(i)
			if !f.IsExported() {
				continue
			}
			if blobTypes[v.Type()] && f.Name == "Data" {
				c.Field(i).Set(r.sentinel(v.Field(i), true))
				continue
			}
			fc, err := r.clone(v.Field(i))
			if err != nil {
				return v, err
			}
			c.Field(i).Set(fc)
		}
		return c, nil
	case reflect.Array:
		c := reflect.New(v.Type()).Elem()
		for i := range v.Len() {
			e, err := r.clone(v.Index(i))
			if err != nil {
				return v, err
			}
			c.Index(i).Set(e)
		}
		return c, nil
	default:
		return v, nil
	}
	// json.RawMessage and the like are JSON, not bytes.
	if v.Kind() == reflect.Slice && v.Type().Elem().Kind() == reflect.Uint8 && !v.Type().Implements(marshalerType) {
		return r.sentinel(v, false), nil
	}
	key := visit{ptr: v.Pointer(), typ: v.Type()}
	if v.Kind() == reflect.Slice {
		key.len = v.Len()
	}
	if r.path[key] {
		return v, fmt.Errorf("llmcapture: redact: cycle via %s", v.Type())
	}
	r.path[key] = true
	defer delete(r.path, key)
	switch v.Kind() {
	case reflect.Pointer:
		e, err := r.clone(v.Elem())
		if err != nil {
			return v, err
		}
		p := reflect.New(v.Type().Elem())
		p.Elem().Set(e)
		return p, nil
	case reflect.Map:
		c := reflect.MakeMapWithSize(v.Type(), v.Len())
		for it := v.MapRange(); it.Next(); {
			e, err := r.clone(it.Value())
			if err != nil {
				return v, err
			}
			c.SetMapIndex(it.Key(), e)
		}
		return c, nil
	default: // reflect.Slice
		c := reflect.MakeSlice(v.Type(), v.Len(), v.Len())
		for i := range v.Len() {
			e, err := r.clone(v.Index(i))
			if err != nil {
				return v, err
			}
			c.Index(i).Set(e)
		}
		return c, nil
	}
}

// sentinel returns a stand-in for the bytes in v, or v itself when empty so
// omitempty still applies.
func (r *redactor) sentinel(v reflect.Value, flatten bool) reflect.Value {
	raw := v.Bytes()
	if len(raw) == 0 {
		return v
	}
	sum := sha256.Sum256(raw)
	s := fmt.Appendf(nil, "llmcapture:%s:%d", r.nonce, len(r.digests))
	r.digests[base64.StdEncoding.EncodeToString(s)] = digest{
		fields:  map[string]any{"sha256": hex.EncodeToString(sum[:]), "size": len(raw)},
		flatten: flatten,
	}
	return reflect.ValueOf(s).Convert(v.Type())
}

// replace swaps each sentinel in node for its digest.
func (r *redactor) replace(node any) any {
	switch n := node.(type) {
	case string:
		if d, ok := r.digests[n]; ok {
			return d.fields
		}
	case map[string]any:
		for k, v := range n {
			if d, ok := r.digests[asString(v)]; ok && d.flatten {
				delete(n, k)
				maps.Copy(n, d.fields)
				continue
			}
			n[k] = r.replace(v)
		}
	case []any:
		for i, v := range n {
			n[i] = r.replace(v)
		}
	}
	return node
}

func asString(v any) string {
	s, _ := v.(string)
	return s
}
