# llmcapture-go

Go implementation of llmcapture (package `llmcapture`):

```sh
go get github.com/pierluc-codes/llmcapture-go
```

Wraps an LLM client so every call is published to NATS as two JSON messages,
`llm.req` before the call and `llm.resp` after it.

```go
gen := llmcapture.Wrap(client.Models, jspub.New(js)).WithAttr("service", "my-service")
resp, err := gen.GenerateContent(ctx, model, contents, cfg)
```

The publisher is anything with `Publish(ctx, subject, msgID, payload)`.

`GenerateContent` has genai's signature, so a `Capturer` drops in for
`*genai.Models`. `Generate` takes the same arguments and returns a `Result`:
the response, the `call_id`, and whether each message was persisted. Unlike the
usual Go convention, the `Result` comes back alongside an error whenever a
`call_id` was assigned, so a failed call can still be traced.

## Options

- `WithSubjects(request, response)`: where the two messages go (default
  `llm.req`, `llm.resp`).
- `WithBestEffortTimeout(d)`: how long a `BestEffort` publish may take (default
  500 ms, also used for `<= 0`). It is short because a slow publish delays the
  call even though it cannot fail it.

Attrs and write policy are set per call site on a copy of the generator:
`gen.WithAttr(key, value)` adds a key/value to both messages' `attrs` (e.g. ids
that join the call to your records), and
`gen.WithPolicy(llmcapture.Policy{Request: llmcapture.Required})` sets the
policy. Both leave `gen` unchanged and chain. Each message is `BestEffort`
(default) or `Required`:

- `BestEffort`: a failed publish is logged with `slog` and the call goes on.
- `Required` request: if it is not persisted, the LLM is not called and
  `GenerateContent` returns an error wrapping `ErrRequestNotPersisted`. Nothing
  was billed, so retrying is safe.
- `Required` response: if it is not persisted, the answer is withheld (`nil`
  plus an error wrapping `ErrResponseNotPersisted`, joined with the call's own
  error if any). The call was billed, so retrying pays again.

Both errors wrap `ErrNotPersisted`.

`Required` is only as strong as `Publish`: it must return nil only once the
message is stored. `jspub.New(js)` is a JetStream publisher that waits for the
PubAck and fails when no stream binds the subject; a core NATS publish returns
before anything is stored.

## Messages

Subjects below are the defaults. Both messages carry `call_id` (UUIDv7); join
on it. Message ids are `<call_id>.req` and `<call_id>.resp`, so JetStream
dedupes a retried publish without dropping one of the pair.

- `llm.req` (`Request`): `call_id`, `started_at`, `model`, `attrs`, `contents`,
  `config`. Published before the call, so a call that never returns (hang,
  crash, OOM kill) still has a record: a request with no response.
- `llm.resp` (`Response`): `call_id`, `model`, `attrs`, `duration_ns`,
  `response`, `error`. Published for every call that returns, failed ones
  included. `model` and `attrs` repeat the request's, so a response can be
  attributed even when its request was lost. `duration_ns` times the LLM call
  only, not the `llm.req` publish.

`contents`, `config` and `response` keep genai's JSON field names. Every byte
field becomes `{sha256, size}` (an `inlineData` blob becomes
`{mimeType, sha256, size}`), so raw bytes (inline media, voice samples, thought
signatures) are never published; the sha256 is of the raw bytes.
`config.httpOptions` (transport settings, which may carry auth headers) and
`response.sdkHttpResponse` (transport headers) are dropped.

Each `Required` publish has a 5 s timeout, each `BestEffort` one the
`WithBestEffortTimeout` value. `llm.req` honors the call's context; `llm.resp`
uses a context detached from it, so a canceled call is still recorded.

## Limits

- **Message size.** NATS rejects messages over its `max_payload` (1 MiB by
  default). A prompt or response that large after redaction (long text; byte
  fields are only a digest) fails to publish every time: under `Required` the
  call fails on every attempt, so retrying does not help. Raise `max_payload`
  or keep those calls `BestEffort`.
- **Timeouts are ambiguous.** A publish that times out may still have been
  stored. Under `Required` the call fails anyway: a stored request then has no
  response, and a stored response's answer is withheld.

## Scope

Only Gemini (`google.golang.org/genai`) `GenerateContent`. Streaming
(`GenerateContentStream`) and other providers are not captured.

Nothing is stored unless a JetStream stream binds the subjects.

## Development

```sh
go test -race ./...
```
