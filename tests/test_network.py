from unittest.mock import patch

import pytest

from areal.utils.network import find_free_ports


def test_find_free_ports_scans_entire_range_with_wraparound():
    checked_ports = []

    def fake_is_port_free(port: int) -> bool:
        checked_ports.append(port)
        return port == 1001

    with (
        patch("areal.utils.network.random.randrange", return_value=2),
        patch("areal.utils.network.is_port_free", side_effect=fake_is_port_free),
    ):
        ports = find_free_ports(1, port_range=(1000, 1004))

    assert ports == [1001]
    assert checked_ports == [1002, 1003, 1004, 1000, 1001]


def test_find_free_ports_respects_excluded_ports_in_range():
    checked_ports = []

    def fake_is_port_free(port: int) -> bool:
        checked_ports.append(port)
        return True

    with (
        patch("areal.utils.network.random.randrange", return_value=0),
        patch("areal.utils.network.is_port_free", side_effect=fake_is_port_free),
    ):
        ports = find_free_ports(
            2,
            port_range=(2000, 2003),
            exclude_ports={1999, 2000, 2001, 3000},
        )

    assert ports == [2002, 2003]
    assert checked_ports == [2002, 2003]


def test_find_free_ports_rejects_invalid_range():
    with pytest.raises(ValueError, match="Invalid port range"):
        find_free_ports(1, port_range=(3000, 2999))
