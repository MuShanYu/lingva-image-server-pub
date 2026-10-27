from pathlib import Path

from app.cpus import available_cpus


def test_available_cpus_takes_affinity_and_cgroup_quota(tmp_path: Path) -> None:
    cpu_max = tmp_path / "cpu.max"
    cpu_max.write_text("max 100000\n")
    assert available_cpus(affinity=8, cpu_max=cpu_max) == 8
    # docker --cpus 2.5：配额向上取整
    cpu_max.write_text("250000 100000\n")
    assert available_cpus(affinity=8, cpu_max=cpu_max) == 3
    cpu_max.write_text("50000 100000\n")
    assert available_cpus(affinity=8, cpu_max=cpu_max) == 1
    assert available_cpus(affinity=4, cpu_max=tmp_path / "missing") == 4
