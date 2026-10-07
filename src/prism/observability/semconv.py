"""Span 属性名的唯一来源（OpenTelemetry 语义约定 + 本项目自定义）。

**为什么这些字符串写字面量、不 import `opentelemetry.semconv`**：GenAI 那批属性还在
`_incubating` 命名空间下（稳定度 Development），路径在版本间挪过。为一个可观测性
功能让整个应用起不来，不划算。属性名本身是**线格式**，规范里就是这几个字符串。

代价是可能漂：`tests/unit/test_tracing.py` 里有一条用例，**装着的 semconv 包能
import 时**逐条断言这些字面量与它一致 —— 标准改了名，用例会红。

设计文档：`acr-min/06-可观测性/21-可观测性设计.md` §3.2 / §3.3
"""

# ---------- GenAI 语义约定（Langfuse 的 OTLP 端点原生识别这些名字）----------

GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read.input_tokens"
GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS = "gen_ai.usage.cache_creation.input_tokens"
GEN_AI_USAGE_REASONING_OUTPUT_TOKENS = "gen_ai.usage.reasoning.output_tokens"

# ---------- 本项目自定义（统一 `prism.` 前缀，不与标准打架）----------

PRISM_GROUP_ID = "prism.group_id"
PRISM_PACKAGE_LABEL = "prism.package_label"
PRISM_HUNK_COUNT = "prism.hunk_count"
PRISM_TOKEN_ESTIMATE = "prism.token_estimate"
PRISM_EXIT_REASON = "prism.exit_reason"
PRISM_ROUND_COUNT = "prism.round_count"
PRISM_PENDING_HUNKS = "prism.pending_hunks"
PRISM_SKIPPED_GROUPS = "prism.skipped_groups"
PRISM_FAILED_GROUPS = "prism.failed_groups"
PRISM_BASE_REV = "prism.base_rev"
PRISM_HEAD_REV = "prism.head_rev"
PRISM_TOOL_NAME = "prism.tool_name"
PRISM_TOOL_ARGUMENTS = "prism.tool_arguments"
PRISM_RESULT_SIZE = "prism.result_size"
PRISM_LINTER_COUNT = "prism.linter_count"
PRISM_FINDING_COUNT = "prism.finding_count"
PRISM_RAW_FINDING_COUNT = "prism.raw_finding_count"
PRISM_SKIP_REASON = "prism.skip_reason"
PRISM_DURATION_MS = "prism.duration_ms"
PRISM_CANDIDATE_COUNT = "prism.candidate_count"
PRISM_REMOVED_COUNT = "prism.removed_count"
PRISM_FAILURE = "prism.failure"
