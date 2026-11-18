package agent

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Autumn-27/norma/llm"
)

type seenRequest struct {
	header http.Header
	body   string
}

// sendThrough drives one request through an llmHTTPClient with the given extras
// and returns what reached the server.
func sendThrough(t *testing.T, extras RequestExtras, method, contentType, body string) seenRequest {
	t.Helper()
	got := make(chan seenRequest, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		if r.ContentLength != int64(len(b)) {
			t.Errorf("Content-Length %d, body %d bytes", r.ContentLength, len(b))
		}
		got <- seenRequest{header: r.Header.Clone(), body: string(b)}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	client, err := llmHTTPClient("", "", "", extras)
	if err != nil {
		t.Fatalf("llmHTTPClient: %v", err)
	}
	var rb io.Reader
	if body != "" {
		rb = strings.NewReader(body)
	}
	req, err := http.NewRequest(method, srv.URL, rb)
	if err != nil {
		t.Fatal(err)
	}
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("%s: %v", method, err)
	}
	resp.Body.Close()
	return <-got
}

func mustExtras(t *testing.T, org, project, body string) RequestExtras {
	t.Helper()
	e, err := parseRequestExtras(org, project, body)
	if err != nil {
		t.Fatalf("parseRequestExtras: %v", err)
	}
	return e
}

func TestRequestExtrasMergeBodyAndHeaders(t *testing.T) {
	e := mustExtras(t, "org-123", "proj_456", `{"access_programs":{"cyber":"daybreak_blue"}}`)
	got := sendThrough(t, e, http.MethodPost, "application/json",
		`{"model":"gpt-6-sol","input":[{"role":"user","content":"hi"}],"stream":true,"store":false}`)

	var body map[string]any
	if err := json.Unmarshal([]byte(got.body), &body); err != nil {
		t.Fatalf("body is not JSON: %v (%s)", err, got.body)
	}
	ap, _ := body["access_programs"].(map[string]any)
	if ap["cyber"] != "daybreak_blue" {
		t.Fatalf("access_programs not merged: %s", got.body)
	}
	// The client's own fields survive untouched.
	if body["model"] != "gpt-6-sol" || body["stream"] != true || body["store"] != false {
		t.Fatalf("original fields changed: %s", got.body)
	}
	if in, _ := body["input"].([]any); len(in) != 1 {
		t.Fatalf("input changed: %s", got.body)
	}
	if got.header.Get("OpenAI-Organization") != "org-123" || got.header.Get("OpenAI-Project") != "proj_456" {
		t.Fatalf("headers: org=%q project=%q", got.header.Get("OpenAI-Organization"), got.header.Get("OpenAI-Project"))
	}
}

func TestRequestExtrasOverwriteSameField(t *testing.T) {
	e := mustExtras(t, "", "", `{"store":true}`)
	got := sendThrough(t, e, http.MethodPost, "application/json", `{"model":"m","store":false}`)
	if !strings.Contains(got.body, `"store":true`) {
		t.Fatalf("extra field should overwrite: %s", got.body)
	}
}

func TestRequestExtrasLeaveNonJSONBodyAlone(t *testing.T) {
	e := mustExtras(t, "", "", `{"access_programs":{"cyber":"daybreak_blue"}}`)
	for _, body := range []string{"a=1&b=2", `["not","an","object"]`} {
		got := sendThrough(t, e, http.MethodPost, "text/plain", body)
		if got.body != body {
			t.Fatalf("non-object body changed: got %q, want %q", got.body, body)
		}
	}
}

func TestRequestExtrasPreserveCallerRequest(t *testing.T) {
	const body = "a=1&b=2"
	const header = "OpenAI-Organization"
	for name, extraBody := range map[string]string{
		"headers only":  "",
		"non-JSON body": `{"store":true}`,
	} {
		t.Run(name, func(t *testing.T) {
			originalBody := io.NopCloser(strings.NewReader(body))
			req, err := http.NewRequest(http.MethodPost, "http://example.test", originalBody)
			if err != nil {
				t.Fatal(err)
			}
			req.Header.Set(header, "org-original")
			transport := quotaAwareTransport{
				extras: mustExtras(t, "org-configured", "", extraBody).compile(),
				base: roundTripperFunc(func(out *http.Request) (*http.Response, error) {
					if got := out.Header.Get(header); got != "org-configured" {
						t.Errorf("outgoing header = %q, want org-configured", got)
					}
					got, err := io.ReadAll(out.Body)
					out.Body.Close()
					if err != nil || string(got) != body {
						t.Errorf("outgoing body = %q, err = %v, want %q", got, err, body)
					}
					return &http.Response{StatusCode: http.StatusOK, Body: http.NoBody}, nil
				}),
			}
			resp, err := transport.RoundTrip(req)
			if err != nil {
				t.Fatal(err)
			}
			resp.Body.Close()
			if got := req.Header.Get(header); got != "org-original" {
				t.Errorf("caller header changed to %q", got)
			}
			if req.Body != originalBody || req.ContentLength != 0 || req.GetBody != nil {
				t.Error("caller body fields changed")
			}
		})
	}
}

func TestRequestExtrasZeroChangesNothing(t *testing.T) {
	const body = `{"model":"m","stream":true}`
	got := sendThrough(t, RequestExtras{}, http.MethodPost, "application/json", body)
	if got.body != body {
		t.Fatalf("body changed without extras: %q", got.body)
	}
	if got.header.Get("OpenAI-Organization") != "" || got.header.Get("OpenAI-Project") != "" {
		t.Fatal("headers set without extras")
	}
}

func TestRequestExtrasSkipGet(t *testing.T) {
	e := mustExtras(t, "org-1", "", `{"access_programs":{"cyber":"daybreak_blue"}}`)
	got := sendThrough(t, e, http.MethodGet, "", "")
	if got.body != "" {
		t.Fatalf("GET grew a body: %q", got.body)
	}
	if got.header.Get("OpenAI-Organization") != "org-1" {
		t.Fatal("headers should still be set on GET")
	}
}

func TestParseRequestExtrasRejectsBadInput(t *testing.T) {
	for name, tc := range map[string][3]string{
		"not json":        {"", "", "access_programs=daybreak"},
		"json array":      {"", "", `["a"]`},
		"json null":       {"", "", `null`},
		"reserved model":  {"", "", `{"model":"other"}`},
		"reserved input":  {"", "", `{"input":[]}`},
		"header newline":  {"org-1\r\nX-Injected: 1", "", ""},
		"header too long": {strings.Repeat("a", MaxUserAgentLen+1), "", ""},
		"body too long":   {"", "", `{"a":"` + strings.Repeat("x", maxExtraBodyLen) + `"}`},
	} {
		if _, err := parseRequestExtras(tc[0], tc[1], tc[2]); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
	e, err := parseRequestExtras("  ", "", "  ")
	if err != nil || !e.IsZero() {
		t.Fatalf("blank values: extras=%+v err=%v, want zero and nil", e, err)
	}
}

func TestFromEnvRequestExtrasOnlyForOpenAIFormats(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "sk-ant")
	t.Setenv("OPENAI_API_KEY", "sk-oai")
	t.Setenv("ARTEX_LLM_BASE_URL", "")
	t.Setenv("ARTEX_LLM_MODEL", "")
	t.Setenv(envOpenAIOrg, "org-1")
	t.Setenv(envOpenAIProject, "proj_1")
	t.Setenv(envExtraBody, `{"access_programs":{"cyber":"daybreak_blue"}}`)

	for _, prov := range []string{"openai", "openai-responses"} {
		t.Setenv("ARTEX_LLM_PROVIDER", prov)
		c, ok := FromEnv()
		if !ok {
			t.Fatalf("%s: FromEnv ok=false", prov)
		}
		if c.Extras.Organization != "org-1" || c.Extras.Project != "proj_1" || c.Extras.Body != `{"access_programs":{"cyber":"daybreak_blue"}}` {
			t.Fatalf("%s: extras not loaded: %+v", prov, c.Extras)
		}
	}
	t.Setenv("ARTEX_LLM_PROVIDER", "anthropic")
	c, ok := FromEnv()
	if !ok || c.Format != llm.FormatAnthropic || !c.Extras.IsZero() {
		t.Fatalf("anthropic must not get OpenAI extras: ok=%v extras=%+v", ok, c.Extras)
	}

	// Config must stay comparable: server.applyLLM compares configs with ==.
	t.Setenv("ARTEX_LLM_PROVIDER", "openai-responses")
	a, _ := FromEnv()
	b, _ := FromEnv()
	if a != b {
		t.Fatal("two FromEnv configs with the same environment must compare equal")
	}

	// An invalid value disables the env provider instead of silently dropping it.
	t.Setenv("ARTEX_LLM_PROVIDER", "openai-responses")
	t.Setenv(envExtraBody, `not json`)
	if _, ok := FromEnv(); ok {
		t.Fatal("invalid ARTEX_LLM_EXTRA_BODY must make FromEnv fail")
	}
}
