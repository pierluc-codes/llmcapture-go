package llmcapture_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/genai"

	"github.com/pierluc-codes/llmcapture-go"
)

var _ llmcapture.Generator = (*genai.Models)(nil)

type stubGenerator struct {
	resp *genai.GenerateContentResponse
	err  error
	// published is how many messages were out when the call ran.
	published int
	pub       *recordingPublisher
	called    bool
}

func (s *stubGenerator) GenerateContent(ctx context.Context, model string, contents []*genai.Content, cfg *genai.GenerateContentConfig) (*genai.GenerateContentResponse, error) {
	s.called = true
	if s.pub != nil {
		s.published = len(s.pub.msgs)
	}
	return s.resp, s.err
}

type published struct {
	ctxErr error
	// timeout is how long the publish had until its deadline.
	timeout time.Duration
	subject string
	msgID   string
	payload []byte
}

type recordingPublisher struct {
	msgs []published
	err  error
}

func (p *recordingPublisher) Publish(ctx context.Context, subject, msgID string, payload []byte) error {
	deadline, _ := ctx.Deadline()
	p.msgs = append(p.msgs, published{ctxErr: ctx.Err(), timeout: time.Until(deadline), subject: subject, msgID: msgID, payload: payload})
	return p.err
}

type requestMsg struct {
	CallID    string            `json:"call_id"`
	StartedAt time.Time         `json:"started_at"`
	Model     string            `json:"model"`
	Attrs     map[string]string `json:"attrs"`
	Contents  []map[string]any  `json:"contents"`
	Config    map[string]any    `json:"config"`
}

type responseMsg struct {
	CallID     string         `json:"call_id"`
	DurationNS int64          `json:"duration_ns"`
	Response   map[string]any `json:"response"`
	Error      string         `json:"error"`
}

func decode[T any](t *testing.T, payload []byte) T {
	t.Helper()
	var v T
	require.NoError(t, json.Unmarshal(payload, &v))
	return v
}

func TestWrapPublishesRequestBeforeCallAndResponseAfter(t *testing.T) {
	resp := &genai.GenerateContentResponse{
		ModelVersion: "gemini-x-001",
		Candidates:   []*genai.Candidate{{Content: genai.NewContentFromText(`{"title":"T"}`, genai.RoleModel)}},
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount: 10, CandidatesTokenCount: 3, TotalTokenCount: 13,
		},
		SDKHTTPResponse: &genai.HTTPResponse{Headers: map[string][]string{"X-Debug": {"1"}}},
	}
	pub := &recordingPublisher{}
	stub := &stubGenerator{resp: resp, pub: pub}
	gen := llmcapture.Wrap(stub, pub).WithAttr("service", "svc").WithAttr("job_id", "job_1")

	ctx := context.Background()
	cfg := &genai.GenerateContentConfig{
		SystemInstruction: genai.NewContentFromText("be brief", genai.RoleUser),
		Temperature:       genai.Ptr[float32](0),
	}
	before := time.Now()
	got, err := gen.GenerateContent(ctx, "gemini-x", []*genai.Content{genai.NewContentFromText("hi", genai.RoleUser)}, cfg)

	require.NoError(t, err)
	assert.Same(t, resp, got)
	assert.Equal(t, 1, stub.published, "llm.req must be out before the call")
	require.Len(t, pub.msgs, 2)

	reqMsg, respMsg := pub.msgs[0], pub.msgs[1]
	assert.Equal(t, llmcapture.SubjectRequest, reqMsg.subject)
	assert.Equal(t, llmcapture.SubjectResponse, respMsg.subject)
	assert.Equal(t, "llm.req", llmcapture.SubjectRequest)
	assert.Equal(t, "llm.resp", llmcapture.SubjectResponse)

	req := decode[requestMsg](t, reqMsg.payload)
	assert.NotEmpty(t, req.CallID)
	assert.Equal(t, req.CallID+".req", reqMsg.msgID)
	assert.WithinDuration(t, before, req.StartedAt, time.Second)
	assert.Equal(t, "gemini-x", req.Model)
	assert.Equal(t, map[string]string{"service": "svc", "job_id": "job_1"}, req.Attrs)
	assert.Equal(t, "hi", req.Contents[0]["parts"].([]any)[0].(map[string]any)["text"])
	assert.Equal(t, "be brief", req.Config["systemInstruction"].(map[string]any)["parts"].([]any)[0].(map[string]any)["text"])

	res := decode[responseMsg](t, respMsg.payload)
	assert.Equal(t, req.CallID, res.CallID)
	assert.Equal(t, req.CallID+".resp", respMsg.msgID)
	assert.GreaterOrEqual(t, res.DurationNS, int64(0))
	assert.Equal(t, "gemini-x-001", res.Response["modelVersion"])
	assert.EqualValues(t, 13, res.Response["usageMetadata"].(map[string]any)["totalTokenCount"])
	assert.NotContains(t, res.Response, "sdkHttpResponse")
	assert.Empty(t, res.Error)
}

func TestWrapReplacesInlineDataWithDigest(t *testing.T) {
	pdf := []byte("%PDF-1.7 not really")
	sum := sha256.Sum256(pdf)
	pub := &recordingPublisher{}
	gen := llmcapture.Wrap(&stubGenerator{resp: &genai.GenerateContentResponse{}}, pub)
	contents := []*genai.Content{genai.NewContentFromParts([]*genai.Part{genai.NewPartFromBytes(pdf, "application/pdf")}, genai.RoleUser)}

	_, err := gen.GenerateContent(context.Background(), "m", contents, nil)

	require.NoError(t, err)
	assert.Equal(t, pdf, contents[0].Parts[0].InlineData.Data, "caller's contents must not be modified")
	inline := decode[requestMsg](t, pub.msgs[0].payload).Contents[0]["parts"].([]any)[0].(map[string]any)["inlineData"].(map[string]any)
	assert.Equal(t, map[string]any{
		"mimeType": "application/pdf",
		"sha256":   hex.EncodeToString(sum[:]),
		"size":     float64(len(pdf)),
	}, inline)
}

func TestWrapRecordsError(t *testing.T) {
	callErr := errors.New("quota exceeded")
	pub := &recordingPublisher{}
	gen := llmcapture.Wrap(&stubGenerator{err: callErr}, pub)

	got, err := gen.GenerateContent(context.Background(), "m", nil, nil)

	assert.Nil(t, got)
	assert.Same(t, callErr, err)
	require.Len(t, pub.msgs, 2)
	res := decode[responseMsg](t, pub.msgs[1].payload)
	assert.Equal(t, "quota exceeded", res.Error)
	assert.Nil(t, res.Response)
}

func TestWrapIgnoresPublishFailure(t *testing.T) {
	resp := &genai.GenerateContentResponse{ModelVersion: "v"}
	pub := &recordingPublisher{err: errors.New("nats down")}
	gen := llmcapture.Wrap(&stubGenerator{resp: resp}, pub)

	got, err := gen.GenerateContent(context.Background(), "m", nil, nil)

	require.NoError(t, err)
	assert.Same(t, resp, got)
	assert.Len(t, pub.msgs, 2, "a failed llm.req must not skip llm.resp")
}

func TestWrapPublishesResponseWhenCallContextCanceled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	pub := &recordingPublisher{}
	gen := llmcapture.Wrap(&stubGenerator{err: context.Canceled}, pub)

	_, _ = gen.GenerateContent(ctx, "m", nil, nil)

	require.Len(t, pub.msgs, 2)
	assert.NoError(t, pub.msgs[1].ctxErr)
}

func TestWithSubjects(t *testing.T) {
	pub := &recordingPublisher{}
	gen := llmcapture.Wrap(&stubGenerator{resp: &genai.GenerateContentResponse{}}, pub,
		llmcapture.WithSubjects("acme.llm.request", "acme.llm.response"))

	_, err := gen.GenerateContent(context.Background(), "m", nil, nil)

	require.NoError(t, err)
	require.Len(t, pub.msgs, 2)
	assert.Equal(t, "acme.llm.request", pub.msgs[0].subject)
	assert.Equal(t, "acme.llm.response", pub.msgs[1].subject)
}

// failingOn fails only the publishes to one subject.
type failingOn struct {
	recordingPublisher
	subject string
}

func (p *failingOn) Publish(ctx context.Context, subject, msgID string, payload []byte) error {
	_ = p.recordingPublisher.Publish(ctx, subject, msgID, payload)
	if subject == p.subject {
		return errors.New("nats down")
	}
	return nil
}

var (
	requiredRequest  = llmcapture.Policy{Request: llmcapture.Required}
	requiredResponse = llmcapture.Policy{Response: llmcapture.Required}
)

func TestRequiredRequestBlocksCallWhenNotPersisted(t *testing.T) {
	pub := &failingOn{subject: llmcapture.SubjectRequest}
	stub := &stubGenerator{resp: &genai.GenerateContentResponse{}}
	gen := llmcapture.Wrap(stub, pub).WithPolicy(requiredRequest)

	got, err := gen.GenerateContent(context.Background(), "m", nil, nil)

	assert.Nil(t, got)
	require.ErrorIs(t, err, llmcapture.ErrNotPersisted)
	assert.ErrorContains(t, err, "nats down")
	assert.False(t, stub.called, "the LLM must not be called")
	assert.Len(t, pub.msgs, 1, "no llm.resp for a call that never ran")
}

func TestRequiredRequestCallsWhenPersisted(t *testing.T) {
	resp := &genai.GenerateContentResponse{ModelVersion: "v"}
	gen := llmcapture.Wrap(&stubGenerator{resp: resp}, &failingOn{subject: llmcapture.SubjectResponse}).WithPolicy(requiredRequest)

	got, err := gen.GenerateContent(context.Background(), "m", nil, nil)

	require.NoError(t, err, "llm.resp is best effort by default")
	assert.Same(t, resp, got)
}

func TestRequiredResponseWithholdsAnswerWhenNotPersisted(t *testing.T) {
	gen := llmcapture.Wrap(&stubGenerator{resp: &genai.GenerateContentResponse{ModelVersion: "v"}},
		&failingOn{subject: llmcapture.SubjectResponse}).WithPolicy(requiredResponse)

	got, err := gen.GenerateContent(context.Background(), "m", nil, nil)

	assert.Nil(t, got)
	require.ErrorIs(t, err, llmcapture.ErrNotPersisted)
}

func TestRequiredResponseKeepsCallError(t *testing.T) {
	callErr := errors.New("quota exceeded")
	gen := llmcapture.Wrap(&stubGenerator{err: callErr}, &failingOn{subject: llmcapture.SubjectResponse}).WithPolicy(requiredResponse)

	_, err := gen.GenerateContent(context.Background(), "m", nil, nil)

	assert.ErrorIs(t, err, callErr)
	assert.ErrorIs(t, err, llmcapture.ErrNotPersisted)
}

func TestRequiredResponseReturnsAnswerWhenPersisted(t *testing.T) {
	resp := &genai.GenerateContentResponse{ModelVersion: "v"}
	gen := llmcapture.Wrap(&stubGenerator{resp: resp}, &recordingPublisher{}).
		WithPolicy(llmcapture.Policy{Request: llmcapture.Required, Response: llmcapture.Required})

	got, err := gen.GenerateContent(context.Background(), "m", nil, nil)

	require.NoError(t, err)
	assert.Same(t, resp, got)
}

func TestWithPolicyLeavesOriginalBestEffort(t *testing.T) {
	resp := &genai.GenerateContentResponse{ModelVersion: "v"}
	gen := llmcapture.Wrap(&stubGenerator{resp: resp}, &failingOn{subject: llmcapture.SubjectResponse})

	_, strictErr := gen.WithPolicy(requiredResponse).GenerateContent(context.Background(), "m", nil, nil)
	got, err := gen.GenerateContent(context.Background(), "m", nil, nil)

	require.ErrorIs(t, strictErr, llmcapture.ErrNotPersisted)
	require.NoError(t, err)
	assert.Same(t, resp, got)
}

func TestWithPolicyKeepsSubjects(t *testing.T) {
	pub := &recordingPublisher{}
	gen := llmcapture.Wrap(&stubGenerator{resp: &genai.GenerateContentResponse{}}, pub,
		llmcapture.WithSubjects("a.req", "a.resp")).WithPolicy(requiredRequest)

	_, err := gen.GenerateContent(context.Background(), "m", nil, nil)

	require.NoError(t, err)
	require.Len(t, pub.msgs, 2)
	assert.Equal(t, "a.req", pub.msgs[0].subject)
	assert.Equal(t, "a.resp", pub.msgs[1].subject)
}

func TestWithAttrLeavesOriginalUnchanged(t *testing.T) {
	pub := &recordingPublisher{}
	base := llmcapture.Wrap(&stubGenerator{resp: &genai.GenerateContentResponse{}}, pub).WithAttr("service", "svc")
	a := base.WithAttr("job_id", "a")
	b := base.WithAttr("job_id", "b").WithPolicy(requiredRequest)

	for _, g := range []llmcapture.Generator{base, a, b} {
		_, err := g.GenerateContent(context.Background(), "m", nil, nil)
		require.NoError(t, err)
	}

	require.Len(t, pub.msgs, 6)
	assert.Equal(t, map[string]string{"service": "svc"}, decode[requestMsg](t, pub.msgs[0].payload).Attrs)
	assert.Equal(t, map[string]string{"service": "svc", "job_id": "a"}, decode[requestMsg](t, pub.msgs[2].payload).Attrs)
	assert.Equal(t, map[string]string{"service": "svc", "job_id": "b"}, decode[requestMsg](t, pub.msgs[4].payload).Attrs)
}

func TestWrapDropsHTTPOptionsFromConfig(t *testing.T) {
	pub := &recordingPublisher{}
	gen := llmcapture.Wrap(&stubGenerator{resp: &genai.GenerateContentResponse{}}, pub)
	cfg := &genai.GenerateContentConfig{
		Temperature: genai.Ptr[float32](0),
		HTTPOptions: &genai.HTTPOptions{Headers: http.Header{"Authorization": {"Bearer secret"}}},
	}

	_, err := gen.GenerateContent(context.Background(), "m", nil, cfg)

	require.NoError(t, err)
	assert.NotContains(t, string(pub.msgs[0].payload), "secret")
	assert.NotContains(t, decode[requestMsg](t, pub.msgs[0].payload).Config, "httpOptions")
	assert.NotNil(t, cfg.HTTPOptions, "caller's config must not be modified")
}

func TestResponseCarriesModelAndAttrs(t *testing.T) {
	pub := &recordingPublisher{}
	gen := llmcapture.Wrap(&stubGenerator{resp: &genai.GenerateContentResponse{}}, pub).WithAttr("job_id", "job_1")

	_, err := gen.GenerateContent(context.Background(), "gemini-x", nil, nil)

	require.NoError(t, err)
	var res struct {
		Model string            `json:"model"`
		Attrs map[string]string `json:"attrs"`
	}
	require.NoError(t, json.Unmarshal(pub.msgs[1].payload, &res))
	assert.Equal(t, "gemini-x", res.Model)
	assert.Equal(t, map[string]string{"job_id": "job_1"}, res.Attrs)
}

func TestNotPersistedErrorsTellWhetherTheCallRan(t *testing.T) {
	ctx := context.Background()
	_, reqErr := llmcapture.Wrap(&stubGenerator{resp: &genai.GenerateContentResponse{}}, &failingOn{subject: llmcapture.SubjectRequest}).
		WithPolicy(requiredRequest).GenerateContent(ctx, "m", nil, nil)
	_, respErr := llmcapture.Wrap(&stubGenerator{resp: &genai.GenerateContentResponse{}}, &failingOn{subject: llmcapture.SubjectResponse}).
		WithPolicy(requiredResponse).GenerateContent(ctx, "m", nil, nil)

	assert.ErrorIs(t, reqErr, llmcapture.ErrRequestNotPersisted)
	assert.NotErrorIs(t, reqErr, llmcapture.ErrResponseNotPersisted)
	assert.ErrorIs(t, reqErr, llmcapture.ErrNotPersisted)
	assert.ErrorIs(t, respErr, llmcapture.ErrResponseNotPersisted)
	assert.NotErrorIs(t, respErr, llmcapture.ErrRequestNotPersisted)
	assert.ErrorIs(t, respErr, llmcapture.ErrNotPersisted)
}

// slowPublisher makes every publish take d.
type slowPublisher struct {
	recordingPublisher
	d time.Duration
}

func (p *slowPublisher) Publish(ctx context.Context, subject, msgID string, payload []byte) error {
	time.Sleep(p.d)
	return p.recordingPublisher.Publish(ctx, subject, msgID, payload)
}

func TestDurationExcludesRequestPublish(t *testing.T) {
	pub := &slowPublisher{d: 200 * time.Millisecond}
	gen := llmcapture.Wrap(&stubGenerator{resp: &genai.GenerateContentResponse{}}, pub)

	_, err := gen.GenerateContent(context.Background(), "m", nil, nil)

	require.NoError(t, err)
	res := decode[responseMsg](t, pub.msgs[1].payload)
	assert.Less(t, time.Duration(res.DurationNS), 100*time.Millisecond)
}

func TestGenerateReturnsCallIDAndPersisted(t *testing.T) {
	resp := &genai.GenerateContentResponse{ModelVersion: "v"}
	pub := &recordingPublisher{}

	r, err := llmcapture.Wrap(&stubGenerator{resp: resp}, pub).Generate(context.Background(), "m", nil, nil)

	require.NoError(t, err)
	assert.Same(t, resp, r.Response)
	assert.Equal(t, decode[requestMsg](t, pub.msgs[0].payload).CallID, r.CallID)
	assert.True(t, r.RequestPersisted)
	assert.True(t, r.ResponsePersisted)
}

func TestGenerateReportsBestEffortFailures(t *testing.T) {
	resp := &genai.GenerateContentResponse{ModelVersion: "v"}

	r, err := llmcapture.Wrap(&stubGenerator{resp: resp}, &failingOn{subject: llmcapture.SubjectResponse}).
		Generate(context.Background(), "m", nil, nil)

	require.NoError(t, err)
	assert.Same(t, resp, r.Response)
	assert.True(t, r.RequestPersisted)
	assert.False(t, r.ResponsePersisted)
}

func TestGenerateReturnsResultWithCallError(t *testing.T) {
	callErr := errors.New("quota exceeded")

	r, err := llmcapture.Wrap(&stubGenerator{err: callErr}, &recordingPublisher{}).Generate(context.Background(), "m", nil, nil)

	assert.Same(t, callErr, err)
	require.NotNil(t, r)
	assert.NotEmpty(t, r.CallID)
	assert.Nil(t, r.Response)
	assert.True(t, r.RequestPersisted)
	assert.True(t, r.ResponsePersisted)
}

func TestGenerateReturnsResultWhenRequiredRequestFails(t *testing.T) {
	r, err := llmcapture.Wrap(&stubGenerator{resp: &genai.GenerateContentResponse{}}, &failingOn{subject: llmcapture.SubjectRequest}).
		WithPolicy(requiredRequest).Generate(context.Background(), "m", nil, nil)

	require.ErrorIs(t, err, llmcapture.ErrRequestNotPersisted)
	require.NotNil(t, r)
	assert.NotEmpty(t, r.CallID)
	assert.False(t, r.RequestPersisted)
	assert.False(t, r.ResponsePersisted)
	assert.Nil(t, r.Response)
}

func TestGenerateReturnsResultWhenRequiredResponseFails(t *testing.T) {
	r, err := llmcapture.Wrap(&stubGenerator{resp: &genai.GenerateContentResponse{ModelVersion: "v"}}, &failingOn{subject: llmcapture.SubjectResponse}).
		WithPolicy(requiredResponse).Generate(context.Background(), "m", nil, nil)

	require.ErrorIs(t, err, llmcapture.ErrResponseNotPersisted)
	require.NotNil(t, r)
	assert.NotEmpty(t, r.CallID)
	assert.True(t, r.RequestPersisted)
	assert.False(t, r.ResponsePersisted)
	assert.Nil(t, r.Response, "the answer is withheld")
}

func digestOf(b []byte) map[string]any {
	sum := sha256.Sum256(b)
	return map[string]any{"sha256": hex.EncodeToString(sum[:]), "size": float64(len(b))}
}

func TestWrapRedactsEveryByteField(t *testing.T) {
	sample, consent, img, fnImg := []byte("voice sample"), []byte("consent"), []byte("png bytes"), []byte("fn png")
	pub := &recordingPublisher{}
	resp := &genai.GenerateContentResponse{Candidates: []*genai.Candidate{{
		Content: genai.NewContentFromParts([]*genai.Part{genai.NewPartFromBytes(img, "image/png")}, genai.RoleModel),
	}}}
	gen := llmcapture.Wrap(&stubGenerator{resp: resp}, pub)
	cfg := &genai.GenerateContentConfig{SpeechConfig: &genai.SpeechConfig{VoiceConfig: &genai.VoiceConfig{
		ReplicatedVoiceConfig: &genai.ReplicatedVoiceConfig{MIMEType: "audio/wav", VoiceSampleAudio: sample, ConsentAudio: consent},
	}}}
	contents := []*genai.Content{genai.NewContentFromParts([]*genai.Part{{FunctionResponse: &genai.FunctionResponse{
		Name:  "f",
		Parts: []*genai.FunctionResponsePart{genai.NewFunctionResponsePartFromBytes(fnImg, "image/png")},
	}}}, genai.RoleUser)}

	_, err := gen.GenerateContent(context.Background(), "m", contents, cfg)

	require.NoError(t, err)
	assert.Equal(t, sample, cfg.SpeechConfig.VoiceConfig.ReplicatedVoiceConfig.VoiceSampleAudio, "caller's config must not be modified")
	assert.Equal(t, img, resp.Candidates[0].Content.Parts[0].InlineData.Data, "caller's response must not be modified")
	req := decode[requestMsg](t, pub.msgs[0].payload)
	voice := req.Config["speechConfig"].(map[string]any)["voiceConfig"].(map[string]any)["replicatedVoiceConfig"].(map[string]any)
	assert.Equal(t, map[string]any{"mimeType": "audio/wav", "voiceSampleAudio": digestOf(sample), "consentAudio": digestOf(consent)}, voice)
	fn := req.Contents[0]["parts"].([]any)[0].(map[string]any)["functionResponse"].(map[string]any)
	assert.Equal(t, map[string]any{"mimeType": "image/png", "sha256": digestOf(fnImg)["sha256"], "size": float64(len(fnImg))},
		fn["parts"].([]any)[0].(map[string]any)["inlineData"])
	out := decode[responseMsg](t, pub.msgs[1].payload).Response["candidates"].([]any)[0].(map[string]any)["content"].(map[string]any)
	assert.Equal(t, map[string]any{"mimeType": "image/png", "sha256": digestOf(img)["sha256"], "size": float64(len(img))},
		out["parts"].([]any)[0].(map[string]any)["inlineData"])
}

func TestWrapKeepsSchemaPropertyNamedInlineData(t *testing.T) {
	pub := &recordingPublisher{}
	gen := llmcapture.Wrap(&stubGenerator{resp: &genai.GenerateContentResponse{}}, pub)
	schema := map[string]any{"type": "object", "properties": map[string]any{
		"inlineData": map[string]any{"type": "object", "properties": map[string]any{"data": map[string]any{"type": "string"}}},
	}}
	_, err := gen.GenerateContent(context.Background(), "m", nil, &genai.GenerateContentConfig{ResponseJsonSchema: schema})

	require.NoError(t, err)
	got := decode[requestMsg](t, pub.msgs[0].payload).Config["responseJsonSchema"]
	assert.Equal(t, schema, got)
}

func TestWrapKeepsLargeIntegers(t *testing.T) {
	pub := &recordingPublisher{}
	gen := llmcapture.Wrap(&stubGenerator{resp: &genai.GenerateContentResponse{}}, pub)
	_, err := gen.GenerateContent(context.Background(), "m", nil,
		&genai.GenerateContentConfig{ResponseJsonSchema: map[string]any{"maximum": int64(1<<53 + 1)}})

	require.NoError(t, err)
	assert.Contains(t, string(pub.msgs[0].payload), `"maximum":9007199254740993`)
}

func TestRequestPublishHonorsCallerDeadline(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	pub := &recordingPublisher{}
	gen := llmcapture.Wrap(&stubGenerator{err: context.Canceled}, pub)

	_, _ = gen.GenerateContent(ctx, "m", nil, nil)

	require.Len(t, pub.msgs, 2)
	assert.ErrorIs(t, pub.msgs[0].ctxErr, context.Canceled)
}

func TestPublishTimeoutFollowsPolicy(t *testing.T) {
	pub := &recordingPublisher{}
	gen := llmcapture.Wrap(&stubGenerator{resp: &genai.GenerateContentResponse{}}, pub)

	_, err := gen.WithPolicy(requiredResponse).GenerateContent(context.Background(), "m", nil, nil)

	require.NoError(t, err)
	require.Len(t, pub.msgs, 2)
	assert.InDelta(t, 500*time.Millisecond, pub.msgs[0].timeout, float64(100*time.Millisecond), "BestEffort request")
	assert.InDelta(t, 5*time.Second, pub.msgs[1].timeout, float64(100*time.Millisecond), "Required response")
}

func TestWithBestEffortTimeout(t *testing.T) {
	pub := &recordingPublisher{}
	gen := llmcapture.Wrap(&stubGenerator{resp: &genai.GenerateContentResponse{}}, pub,
		llmcapture.WithBestEffortTimeout(2*time.Second)).WithPolicy(requiredRequest)

	_, err := gen.GenerateContent(context.Background(), "m", nil, nil)

	require.NoError(t, err)
	require.Len(t, pub.msgs, 2)
	assert.InDelta(t, 5*time.Second, pub.msgs[0].timeout, float64(100*time.Millisecond), "Required request")
	assert.InDelta(t, 2*time.Second, pub.msgs[1].timeout, float64(100*time.Millisecond), "BestEffort response")
}

func TestNonPositiveBestEffortTimeoutUsesDefault(t *testing.T) {
	for _, d := range []time.Duration{0, -time.Second} {
		pub := &recordingPublisher{}
		gen := llmcapture.Wrap(&stubGenerator{resp: &genai.GenerateContentResponse{}}, pub, llmcapture.WithBestEffortTimeout(d))

		_, err := gen.GenerateContent(context.Background(), "m", nil, nil)

		require.NoError(t, err)
		require.Len(t, pub.msgs, 2)
		for _, m := range pub.msgs {
			assert.InDelta(t, llmcapture.DefaultBestEffortTimeout, m.timeout, float64(100*time.Millisecond), "WithBestEffortTimeout(%s)", d)
		}
	}
}

func TestWrapRejectsCyclicValue(t *testing.T) {
	schema := map[string]any{}
	schema["self"] = schema
	pub := &recordingPublisher{}
	stub := &stubGenerator{resp: &genai.GenerateContentResponse{}}
	gen := llmcapture.Wrap(stub, pub).WithPolicy(requiredRequest)

	_, err := gen.GenerateContent(context.Background(), "m", nil, &genai.GenerateContentConfig{ResponseJsonSchema: schema})

	require.ErrorIs(t, err, llmcapture.ErrRequestNotPersisted)
	assert.ErrorContains(t, err, "cycle")
	assert.False(t, stub.called)
}

func TestWrapKeepsSharedValues(t *testing.T) {
	shared := map[string]any{"type": "string"}
	pub := &recordingPublisher{}
	gen := llmcapture.Wrap(&stubGenerator{resp: &genai.GenerateContentResponse{}}, pub)

	_, err := gen.GenerateContent(context.Background(), "m", nil,
		&genai.GenerateContentConfig{ResponseJsonSchema: map[string]any{"a": shared, "b": shared}})

	require.NoError(t, err)
	assert.Equal(t, map[string]any{"a": shared, "b": shared}, decode[requestMsg](t, pub.msgs[0].payload).Config["responseJsonSchema"])
}

func TestWrapOmitsEmptyByteFields(t *testing.T) {
	pub := &recordingPublisher{}
	gen := llmcapture.Wrap(&stubGenerator{resp: &genai.GenerateContentResponse{}}, pub)
	contents := []*genai.Content{{Parts: []*genai.Part{{Text: "x", ThoughtSignature: []byte{}}}}}

	_, err := gen.GenerateContent(context.Background(), "m", contents, nil)

	require.NoError(t, err)
	assert.Equal(t, map[string]any{"text": "x"}, decode[requestMsg](t, pub.msgs[0].payload).Contents[0]["parts"].([]any)[0])
}

func TestWrapNestsDigestOfDataOutsideBlobs(t *testing.T) {
	data := []byte("x")
	pub := &recordingPublisher{}
	gen := llmcapture.Wrap(&stubGenerator{resp: &genai.GenerateContentResponse{}}, pub)
	contents := []*genai.Content{{Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
		Name: "f", Args: map[string]any{"data": data, "size": "L"},
	}}}}}

	_, err := gen.GenerateContent(context.Background(), "m", contents, nil)

	require.NoError(t, err)
	part := decode[requestMsg](t, pub.msgs[0].payload).Contents[0]["parts"].([]any)[0].(map[string]any)
	assert.Equal(t, map[string]any{"data": digestOf(data), "size": "L"}, part["functionCall"].(map[string]any)["args"])
}

func TestWrapRedactsBytesInsideArrays(t *testing.T) {
	secret := []byte("secret")
	pub := &recordingPublisher{}
	gen := llmcapture.Wrap(&stubGenerator{resp: &genai.GenerateContentResponse{}}, pub)
	contents := []*genai.Content{{Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
		Name: "f", Args: map[string]any{"a": [1][]byte{secret}},
	}}}}}

	_, err := gen.GenerateContent(context.Background(), "m", contents, nil)

	require.NoError(t, err)
	part := decode[requestMsg](t, pub.msgs[0].payload).Contents[0]["parts"].([]any)[0].(map[string]any)
	assert.Equal(t, map[string]any{"a": []any{digestOf(secret)}}, part["functionCall"].(map[string]any)["args"])
}
