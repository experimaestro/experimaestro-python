"""Web UI URLs: /xps/<experiment>/<view>, /all/<view>, /orphans and the job
detail (see ``app/src/routes.ts``)

The server serves the page for all of them; in the browser, the URL follows
what is shown, back/forward and links open the page they name.
"""

from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from experimaestro.webui.app import create_app

from .scenario import EVAL_DONE, LEARN_DONE, OLD_LEARN
from .web_driver import TOKEN


@pytest.fixture
def client(provider):
    server = SimpleNamespace(state_provider=provider, token=TOKEN)
    with TestClient(create_app(server), follow_redirects=False) as client:
        yield client


@pytest.mark.parametrize(
    "path",
    ["/", "/xps/baseline/services", "/xps/a.b/jobs/pkg.Learn/123/logs", "/orphans"],
)
def test_pages_serve_the_ui(client, path):
    client.cookies.set("experimaestro_token", TOKEN)
    response = client.get(path)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert '<script src="/index.js">' in response.text


def test_assets_are_not_pages(client):
    client.cookies.set("experimaestro_token", TOKEN)
    response = client.get("/index.js")
    assert response.headers["content-type"] == "application/javascript"


def test_login_returns_to_the_page(client):
    response = client.get("/xps/a.b/jobs?run=r1")
    assert response.status_code == 302
    assert response.headers["location"] == (
        "/login.html?next=%2Fxps%2Fa.b%2Fjobs%3Frun%3Dr1"
    )

    response = client.get(
        "/auth", params={"xpm-token": TOKEN, "next": "/xps/a.b/jobs?run=r1"}
    )
    assert response.status_code == 302
    assert response.headers["location"] == "/xps/a.b/jobs?run=r1"


@pytest.mark.parametrize("next_page", [None, "https://example.com", "//example.com"])
def test_login_stays_local(client, next_page):
    params = {"xpm-token": TOKEN}
    if next_page is not None:
        params["next"] = next_page
    response = client.get("/auth", params=params)
    assert response.headers["location"] == "/"


# --- In the browser


def _path(ui) -> str:
    return ui.page.url.removeprefix(ui.url)


@pytest.mark.anyio
@pytest.mark.ui("browser")
async def test_url_follows_the_view(ui):
    await ui.open_experiment("baseline")
    assert _path(ui) == "/xps/baseline/jobs"
    await ui._nav("Services")
    assert _path(ui) == "/xps/baseline/services"

    await ui.page.go_back()
    await ui.settle()
    assert _path(ui) == "/xps/baseline/jobs"
    assert LEARN_DONE in await ui.visible_jobs()

    await ui._nav("Experiments")
    assert _path(ui) == "/"


@pytest.mark.anyio
@pytest.mark.ui("browser")
async def test_open_a_run_from_its_url(ui):
    await ui.page.goto(f"{ui.url}/xps/baseline/jobs?run=20260101_100000")
    await ui.settle()
    assert await ui.visible_jobs() == {OLD_LEARN}

    await ui.page.goto(f"{ui.url}/xps/baseline")
    await ui.settle()
    # Normalised to the default view
    assert _path(ui) == "/xps/baseline/jobs"
    assert EVAL_DONE in await ui.visible_jobs()


@pytest.mark.anyio
@pytest.mark.ui("browser")
async def test_open_services_from_their_url(ui):
    await ui.page.goto(f"{ui.url}/xps/sweep/services")
    await ui.settle()
    rows = ui.page.locator("tr[data-service-id]")
    assert {
        await row.get_attribute("data-experiment-id") for row in await rows.all()
    } == {"sweep"}


@pytest.mark.anyio
@pytest.mark.ui("browser")
async def test_job_detail_url(ui, provider):
    task_id = provider.jobs[LEARN_DONE].task_id
    await ui.page.goto(f"{ui.url}/xps/baseline/jobs/{task_id}/{LEARN_DONE}/logs")
    modal = ui.page.locator(".modal.show")
    await modal.wait_for()
    await ui.settle()
    logs = modal.get_by_role("tab", name="Logs", exact=True)
    assert await logs.get_attribute("aria-selected") == "true"

    await modal.get_by_role("tab", name="Details").click()
    await ui.settle()
    assert _path(ui) == f"/xps/baseline/jobs/{task_id}/{LEARN_DONE}"

    await ui.page.keyboard.press("Escape")
    await ui.settle()
    assert _path(ui) == "/xps/baseline/jobs"
    # Back opens the job again
    await ui.page.go_back()
    await modal.wait_for()
