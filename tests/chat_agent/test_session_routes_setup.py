"""Each setup_session_routes call builds its own router."""

from types import SimpleNamespace

from routes.session_routes import setup_session_routes

_CONTEXT_INFO = "/api/session/{session_id}/context_info"


def test_a_second_setup_does_not_inherit_the_first_ones_endpoints():
    # A shared router kept the first caller's closures, so a later caller
    # that looked up an endpoint by path reached another session manager.
    setup_session_routes(SimpleNamespace(), {})
    router = setup_session_routes(SimpleNamespace(), {})

    assert [route.path for route in router.routes].count(_CONTEXT_INFO) == 1
