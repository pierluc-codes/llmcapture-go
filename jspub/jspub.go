// Package jspub is an llmcapture.Publisher on JetStream.
package jspub

import (
	"context"
	"fmt"

	"github.com/nats-io/nats.go/jetstream"
)

// Publisher returns only once the stream has acked the message, which is what
// llmcapture.Required needs.
type Publisher struct{ js jetstream.JetStream }

func New(js jetstream.JetStream) *Publisher { return &Publisher{js: js} }

func (p *Publisher) Publish(ctx context.Context, subject, msgID string, payload []byte) error {
	if _, err := p.js.Publish(ctx, subject, payload, jetstream.WithMsgID(msgID)); err != nil {
		return fmt.Errorf("jetstream.Publish(%s, %s): %w", subject, msgID, err)
	}
	return nil
}
