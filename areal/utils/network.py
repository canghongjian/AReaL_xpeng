import random
import socket


def gethostname():
    return socket.gethostname()


def gethostip(probe_host: str = "8.8.8.8", probe_port: int = 80) -> str:
    """
    Find the local IPv4 address for outbound route to `probe_host:probe_port` (typically
    a LAN/private IP). Use hostname resolution first; if it fails or returns loopback (127.*),
    fall back to a UDP connect.

    Args:
        probe_host: Remote IPv4 address used to trigger route selection, default to Google
                    Public DNS IP.
        probe_port: Remote port used for the UDP probe.

    Returns:
        The selected local IPv4 address as a string

    Raises:
        RuntimeError: If no suitable IPv4 address can be determined
    """
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if ip and not ip.startswith("127."):
            return ip
    except socket.gaierror:
        pass

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect((probe_host, probe_port))
            return sock.getsockname()[0]
    except OSError as e:
        raise RuntimeError("Could not determine host IP") from e


def find_free_ports(
    count: int, port_range: tuple = (1024, 65535), exclude_ports: set[int] | None = None
) -> list[int]:
    """
    Find multiple free ports within a specified range.

    Args:
        count: Number of free ports to find
        port_range: Tuple of (min_port, max_port) to search within
        exclude_ports: Set of ports to exclude from search

    Returns:
        List of free port numbers

    Raises:
        ValueError: If unable to find requested number of free ports
    """
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    if count == 0:
        return []

    if exclude_ports is None:
        exclude_ports = set()

    min_port, max_port = port_range
    if min_port > max_port:
        raise ValueError(f"Invalid port range: {port_range}")

    # Only ports inside the search range reduce effective capacity.
    exclude_ports = {port for port in exclude_ports if min_port <= port <= max_port}
    free_ports = []

    # Calculate available port range
    available_range = max_port - min_port + 1 - len(exclude_ports)

    if count > available_range:
        raise ValueError(
            f"Cannot find {count} ports in range {port_range}. "
            f"Only {available_range} ports available."
        )

    # Use a random scan start to reduce contention across concurrent allocators,
    # then scan the whole range sequentially to avoid random-attempt failures.
    port_count = max_port - min_port + 1
    start_port = min_port + random.randrange(port_count)
    scanned_ports = 0

    for offset in range(port_count):
        port = min_port + ((start_port - min_port + offset) % port_count)
        if port in exclude_ports:
            continue
        scanned_ports += 1
        if is_port_free(port):
            free_ports.append(port)
            if len(free_ports) == count:
                return sorted(free_ports)

    raise ValueError(
        f"Could only find {len(free_ports)} free ports "
        f"out of {count} requested after scanning {scanned_ports} candidate ports"
    )


def is_port_free(port: int) -> bool:
    """
    Check if a port is free by attempting to bind to it.

    Args:
        port: Port number to check

    Returns:
        True if port is free, False otherwise
    """
    # Check TCP
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("", port))
    except OSError:
        return False
    finally:
        sock.close()

    # Check UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()
