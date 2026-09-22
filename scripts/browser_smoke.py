"""Check the upload workflow and capture desktop/mobile screenshots."""

import argparse
import base64
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--channel", default=None, help="Use 'chrome' for an installed Chrome browser"
    )
    parser.add_argument("--screenshots", default="artifacts/browser")
    args = parser.parse_args()
    output = Path(args.screenshots)
    output.mkdir(parents=True, exist_ok=True)
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.channel, headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1040}, device_scale_factor=1)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url)
        expect(page.locator("#service-status")).to_have_text("Service ready")
        expect(page.locator("#run-inference")).to_be_disabled()
        page.screenshot(path=str(output / "empty.png"), full_page=True)

        # File upload, inference, and JSON export.
        page.locator("#file-input").set_input_files("evaluation/images/coffee.png")
        expect(page.locator("#run-inference")).to_be_enabled()
        expect(page.locator("#preview")).to_be_visible()
        page.locator("#run-inference").click()
        expect(page.locator("#prediction-content")).to_be_visible()
        expect(page.locator(".prediction")).to_have_count(3)
        expect(page.locator(".prediction-label").first).to_have_text("espresso")
        expect(page.locator("#inference-time")).to_contain_text("ms")
        with page.expect_download() as download:
            page.locator("#export-json").click()
        download.value.save_as(str(output / "prediction.json"))
        result = json.loads((output / "prediction.json").read_text())
        assert result["predictions"][0]["label"] == "espresso"
        page.screenshot(path=str(output / "playground.png"), full_page=True)

        # Replacing input must clear previous predictions instead of showing stale data.
        page.locator("[data-sample='chelsea']").click()
        expect(page.locator("#file-name")).to_have_text("chelsea.png")
        expect(page.locator("#prediction-content")).to_be_hidden()
        page.locator("#run-inference").click()
        expect(page.locator("#prediction-content")).to_be_visible()
        expect(page.locator(".prediction")).to_have_count(3)
        page.locator("#clear-image").click()
        expect(page.locator("#run-inference")).to_be_disabled()
        expect(page.locator("#preview-wrap")).to_be_hidden()
        expect(page.locator("#result-empty")).to_be_visible()

        # Drag and drop through a browser DataTransfer.
        page.evaluate(
            """(encoded) => {
                const bytes = Uint8Array.from(atob(encoded), c => c.charCodeAt(0));
                const transfer = new DataTransfer();
                transfer.items.add(new File([bytes], 'dropped.png', {type: 'image/png'}));
                document.getElementById('dropzone').dispatchEvent(
                    new DragEvent('drop', {bubbles: true, cancelable: true, dataTransfer: transfer})
                );
            }""",
            base64.b64encode(Path("evaluation/images/coffee.png").read_bytes()).decode(),
        )
        expect(page.locator("#file-name")).to_have_text("dropped.png")
        expect(page.locator("#run-inference")).to_be_enabled()

        # Client validation and a live API error both have usable error states.
        page.locator("#file-input").set_input_files(
            {"name": "bad.txt", "mimeType": "text/plain", "buffer": b"invalid"}
        )
        expect(page.locator("#error-message")).to_contain_text("JPEG")
        expect(page.locator("#run-inference")).to_be_disabled()
        page.locator("#file-input").set_input_files("evaluation/images/coffee.png")
        page.route(
            "**/predict",
            lambda route: route.fulfill(
                status=503,
                content_type="application/json",
                body=json.dumps(
                    {"error": {"code": "model_not_ready", "message": "The model is not ready yet."}}
                ),
            ),
        )
        page.locator("#run-inference").click()
        expect(page.locator("#error-message")).to_contain_text("not ready")
        expect(page.locator("#run-inference")).to_be_enabled()
        page.unroute("**/predict")

        # Model card is keyboard dismissible and populated by backend metadata.
        page.locator(".nav-item[data-dialog='model']").click()
        expect(page.locator("#info-dialog")).to_be_visible()
        expect(page.locator("#dialog-content")).to_contain_text("IMAGENET1K_V1")
        page.keyboard.press("Escape")
        expect(page.locator("#info-dialog")).to_be_hidden()

        page.locator(".nav-item[data-dialog='evaluation']").click()
        expect(page.locator("#info-dialog")).to_be_visible()
        expect(page.locator(".evaluation-table tbody tr")).to_have_count(4)
        expect(page.locator("#dialog-content")).to_contain_text("75%")
        page.keyboard.press("Escape")

        page.locator(".nav-item[data-dialog='rollout']").click()
        page.get_by_role("button", name="Start bad release", exact=True).click()
        expect(page.locator("#dialog-content")).to_contain_text("10%")
        page.get_by_role("button", name="Check and advance", exact=True).click()
        expect(page.locator("#dialog-content")).to_contain_text("rolled back")
        expect(page.locator("#dialog-content")).to_contain_text("failed")
        page.get_by_role("button", name="Start good release", exact=True).click()
        for percent in ["50%", "100%"]:
            page.get_by_role("button", name="Check and advance", exact=True).click()
            expect(page.locator("#dialog-content")).to_contain_text(percent)
        page.get_by_role("button", name="Check and advance", exact=True).click()
        expect(page.locator("#dialog-content")).to_contain_text("promoted")
        page.screenshot(path=str(output / "rollout.png"), full_page=True)
        page.keyboard.press("Escape")

        page.set_viewport_size({"width": 390, "height": 844})
        page.locator("#run-inference").click()
        expect(page.locator("#prediction-content")).to_be_visible()
        expect(page.locator("#run-inference")).to_be_enabled()
        expect(page.locator(".prediction")).to_have_count(3)
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        page.screenshot(path=str(output / "mobile.png"), full_page=True)
        browser.close()
    assert not errors, errors
    print(f"Browser workflow passed. Screenshots: {output}")


if __name__ == "__main__":
    main()
