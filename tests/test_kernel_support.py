from rl_engine.kernels.support import get_linear_logp_support_matrix


def test_linear_logp_support_matrix_is_framework_consumable():
    matrix = get_linear_logp_support_matrix()
    required = {
        "source",
        "backend",
        "implementation",
        "dtype",
        "hardware",
        "tp",
        "cp",
        "entropy",
        "full_gradient",
    }

    assert matrix
    assert all(required <= set(row) for row in matrix)
    assert {row["backend"] for row in matrix} >= {"cuda_sm90", "triton", "pytorch"}
    assert all(row["source"] == "rl_kernel" for row in matrix)
    assert not any("vime" in value.lower() for row in matrix for value in row.values())


def test_linear_logp_support_matrix_returns_copies():
    matrix = get_linear_logp_support_matrix()
    matrix[0]["backend"] = "mutated"

    assert get_linear_logp_support_matrix()[0]["backend"] != "mutated"
