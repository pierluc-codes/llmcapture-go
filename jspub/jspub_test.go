package jspub_test

import (
	"context"
	"testing"
	"time"

	"github.com/nats-io/nats-server/v2/server"
	"github.com/nats-io/nats.go"
	"github.com/nats-io/nats.go/jetstream"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"git.pierluc-codes.com/pierluc-codes/llmcapture-go"
	"git.pierluc-codes.com/pierluc-codes/llmcapture-go/jspub"
)

var _ llmcapture.Publisher = (*jspub.Publisher)(nil)

func startJetStream(t *testing.T) jetstream.JetStream {
	t.Helper()
	srv, err := server.NewServer(&server.Options{Port: -1, JetStream: true, StoreDir: t.TempDir()})
	require.NoError(t, err)
	srv.Start()
	t.Cleanup(srv.Shutdown)
	require.True(t, srv.ReadyForConnections(5*time.Second))
	nc, err := nats.Connect(srv.ClientURL())
	require.NoError(t, err)
	t.Cleanup(nc.Close)
	js, err := jetstream.New(nc)
	require.NoError(t, err)
	return js
}

func TestPublishStoresOnceWithMsgID(t *testing.T) {
	js := startJetStream(t)
	ctx := context.Background()
	stream, err := js.CreateStream(ctx, jetstream.StreamConfig{Name: "LLM", Subjects: []string{"llm.>"}})
	require.NoError(t, err)
	pub := jspub.New(js)

	require.NoError(t, pub.Publish(ctx, "llm.req", "call_1.req", []byte(`{"a":1}`)))
	require.NoError(t, pub.Publish(ctx, "llm.req", "call_1.req", []byte(`{"a":1}`)))

	info, err := stream.Info(ctx)
	require.NoError(t, err)
	assert.EqualValues(t, 1, info.State.Msgs, "a retried publish is deduped on msg id")
	msg, err := stream.GetLastMsgForSubject(ctx, "llm.req")
	require.NoError(t, err)
	assert.Equal(t, []byte(`{"a":1}`), msg.Data)
	assert.Equal(t, "call_1.req", msg.Header.Get(jetstream.MsgIDHeader))
}

// Required relies on this: nothing stored must be an error.
func TestPublishFailsWhenNoStreamBindsSubject(t *testing.T) {
	pub := jspub.New(startJetStream(t))

	err := pub.Publish(context.Background(), "llm.req", "call_1.req", []byte(`{}`))

	assert.Error(t, err)
}
