import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--simulation-only", action="store_true", default=False, help="Run simulation only, it will run test with `simulation` marker in simulation mode"
    )

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "simulation: mark simulation test that can be executed without a NeuronDevice"
    )

@pytest.fixture
def simulation_only(request):
    return request.config.getoption("--simulation-only")

def pytest_collection_modifyitems(session, config, items):
    if config.getoption("--simulation-only"):
        # Only run cases with `simulation marker`
        result = []
        for item in items:
            for marker in item.iter_markers():
                if marker.name == 'simulation':
                    result.append(item)
                    break
        items.clear()
        items.extend(result)
        