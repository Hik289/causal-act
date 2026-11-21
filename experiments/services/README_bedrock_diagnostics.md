# Bedrock response diagnostics

`bedrock_metadata.py` records provider responses only for explicitly selected
workers using `CAUSALACT_DIAGNOSTIC_WORKERS` and `CAUSALACT_DIAGNOSTIC_DIR`.
Files are private (mode 0600), use unique names, and contain the request hash
rather than the prompt or image. Recording cannot trigger another model request;
recording failures leave inference unchanged. Provider text stays in private
evidence and must not be published with experiment results.

The relay records the provider stop reason. A refusal produces HTTP 422 with
`provider_refusal` and `retryable: false`, including when the provider also emits
text. It supplies no action for execution. Other empty completions retain HTTP
502; usable text, including text at a token limit, stays with the native parser.
Neither error represents a benchmark failure or a zero score.

Twelve offline checks include the actual relay request handler with mocked
upstream responses; refusal and normal-response cases each issue one call.

A separately registered synthetic diagnostic on November 21 returned upstream
`stop_reason=refusal`, no content blocks, 275 input tokens and zero output tokens.
It used the original synthetic request without alteration, through the shared
8-inflight / 8-starts-per-6-seconds gateway, on a separate credential binding.
The ordinary backend stayed alive and the original paced frontend was restored.
This identifies that diagnostic response, not the cause of earlier responses
whose upstream metadata was unavailable, and does not establish ordinary-backend
recovery. No benchmark or transfer score was produced. Refusals do not authorize
prompt rewriting or repeated requests to obtain a different answer.

Run the offline checks with:

```sh
python -m unittest discover -s experiments/services -p 'test_bedrock_*.py'
```
