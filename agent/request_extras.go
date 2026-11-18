package agent

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
)

// Environment variables for fixed LLM request extras (OpenAI formats only):
//
//	ARTEX_LLM_OPENAI_ORG     = value of the OpenAI-Organization header
//	ARTEX_LLM_OPENAI_PROJECT = value of the OpenAI-Project header
//	ARTEX_LLM_EXTRA_BODY     = JSON object merged into every request body, e.g.
//	                           {"access_programs":{"cyber":"daybreak_blue"}}
const (
	envOpenAIOrg     = "ARTEX_LLM_OPENAI_ORG"
	envOpenAIProject = "ARTEX_LLM_OPENAI_PROJECT"
	envExtraBody     = "ARTEX_LLM_EXTRA_BODY"
)

// maxExtraBodyLen bounds ARTEX_LLM_EXTRA_BODY: it is added to every request.
const maxExtraBodyLen = 16 << 10

// RequestExtras holds fixed headers and top-level JSON body fields added to every
// LLM request. The zero value changes nothing. It holds only strings so that
// Config stays comparable with == (the server compares configs to reuse providers).
type RequestExtras struct {
	// Organization / Project set the OpenAI-Organization / OpenAI-Project headers.
	Organization string
	Project      string
	// Body is a compact JSON object whose fields are merged into the top level of
	// every JSON-object request body, overwriting a field of the same name — the
	// equivalent of the OpenAI SDKs' extra_body. Empty = none.
	Body string
}

// IsZero reports whether the extras change nothing.
func (e RequestExtras) IsZero() bool { return e == RequestExtras{} }

// RequestExtrasFromEnv reads the request extras from the environment. Unset or
// blank variables contribute nothing; an invalid value is an error.
func RequestExtrasFromEnv() (RequestExtras, error) {
	return parseRequestExtras(os.Getenv(envOpenAIOrg), os.Getenv(envOpenAIProject), os.Getenv(envExtraBody))
}

func parseRequestExtras(org, project, body string) (RequestExtras, error) {
	e := RequestExtras{Organization: strings.TrimSpace(org), Project: strings.TrimSpace(project)}
	if err := checkHeaderValue(e.Organization); err != nil {
		return RequestExtras{}, fmt.Errorf("%s: %w", envOpenAIOrg, err)
	}
	if err := checkHeaderValue(e.Project); err != nil {
		return RequestExtras{}, fmt.Errorf("%s: %w", envOpenAIProject, err)
	}
	body = strings.TrimSpace(body)
	if body == "" {
		return e, nil
	}
	if len(body) > maxExtraBodyLen {
		return RequestExtras{}, fmt.Errorf("%s: too long (%d bytes, max %d)", envExtraBody, len(body), maxExtraBodyLen)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal([]byte(body), &fields); err != nil || fields == nil {
		return RequestExtras{}, fmt.Errorf("%s: must be a JSON object, e.g. {\"access_programs\":{\"cyber\":\"daybreak_blue\"}}", envExtraBody)
	}
	for _, reserved := range []string{"model", "input", "messages", "stream", "tools"} {
		if _, ok := fields[reserved]; ok {
			return RequestExtras{}, fmt.Errorf("%s: field %q is set by ARTEX and cannot be overridden", envExtraBody, reserved)
		}
	}
	if len(fields) > 0 {
		var buf bytes.Buffer
		if err := json.Compact(&buf, []byte(body)); err != nil {
			return RequestExtras{}, fmt.Errorf("%s: %w", envExtraBody, err)
		}
		e.Body = buf.String()
	}
	return e, nil
}

// checkHeaderValue rejects values that could not be sent as one header line.
func checkHeaderValue(v string) error {
	if len(v) > MaxUserAgentLen {
		return fmt.Errorf("too long (%d bytes, max %d)", len(v), MaxUserAgentLen)
	}
	for _, r := range v {
		if r < 0x20 || r == 0x7f {
			return fmt.Errorf("contains a control character")
		}
	}
	return nil
}

// compiledExtras is RequestExtras parsed once for the transport.
type compiledExtras struct {
	headers map[string]string
	body    map[string]json.RawMessage
}

func (e RequestExtras) compile() compiledExtras {
	var c compiledExtras
	for name, v := range map[string]string{"OpenAI-Organization": e.Organization, "OpenAI-Project": e.Project} {
		if v != "" {
			if c.headers == nil {
				c.headers = map[string]string{}
			}
			c.headers[name] = v
		}
	}
	if e.Body != "" {
		// Validated by parseRequestExtras; an unparsable value merges nothing.
		_ = json.Unmarshal([]byte(e.Body), &c.body)
	}
	return c
}

// applyBody returns req with the extra body fields merged into its JSON body. A
// request without extras, without a body, or whose body is not a JSON object is
// returned unchanged (a non-JSON body is restored byte for byte). The original
// request is never modified: a merged body goes on a clone.
func (c compiledExtras) applyBody(req *http.Request) *http.Request {
	if len(c.body) == 0 || req.Method != http.MethodPost || req.Body == nil || req.Body == http.NoBody {
		return req
	}
	orig, err := io.ReadAll(req.Body)
	_ = req.Body.Close()
	if err != nil {
		// The body is consumed and unreadable; let the send fail on what remains.
		setBody(req, orig)
		return req
	}
	var fields map[string]json.RawMessage
	if json.Unmarshal(orig, &fields) != nil || fields == nil {
		setBody(req, orig) // not a JSON object: send it exactly as it was
		return req
	}
	for k, v := range c.body {
		fields[k] = v
	}
	merged, err := json.Marshal(fields)
	if err != nil {
		setBody(req, orig)
		return req
	}
	out := req.Clone(req.Context())
	setBody(out, merged)
	return out
}

func setBody(req *http.Request, b []byte) {
	req.Body = io.NopCloser(bytes.NewReader(b))
	req.ContentLength = int64(len(b))
	req.GetBody = func() (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(b)), nil }
}
