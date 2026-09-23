"""Check the upload workflow and capture desktop/mobile screenshots."""

import argparse
import base64
import json
from pathlib import Path
from time import sleep

from playwright.sync_api import expect, sync_playwright

from canary_vision.model import model_path


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
        expect(page.locator("#rollout-view")).to_be_hidden()
        expect(page.locator("#playground-view")).to_be_visible()
        page.screenshot(path=str(output / "empty.png"), full_page=True, animations="disabled")

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
        page.screenshot(path=str(output / "playground.png"), full_page=True, animations="disabled")

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
        page.locator("#file-input").set_input_files("evaluation/images/coffee.png")

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

        # Automatic presets show traffic, gate evidence, and the final outcome.
        page.locator("#show-rollout").click()
        expect(page.locator("#playground-view")).to_be_hidden()
        expect(page.locator("#rollout-view")).to_be_visible()
        expect(page.locator("#show-rollout")).to_have_attribute("aria-current", "page")
        expect(page.locator(".model-option")).to_have_count(2)
        assert (
            page.locator("#model-options").evaluate("el => getComputedStyle(el).display") == "grid"
        )
        expect(page.locator("#start-rollout")).to_be_enabled()
        page.screenshot(
            path=str(output / "rollout-choice.png"), full_page=True, animations="disabled"
        )

        # Slow status responses must eventually render instead of being discarded.
        def slow_status(route):
            response = route.fetch()
            sleep(1.4)
            route.fulfill(response=response)

        page.route("**/rollout", slow_status, times=1)
        page.reload()
        expect(page.locator("#start-rollout")).to_be_enabled(timeout=10000)
        expect(page.locator("#rollout-connection")).to_be_hidden()
        page.unroute("**/rollout")

        # Connection failures explain the disabled action but allow model selection.
        page.route("**/rollout", lambda route: route.fulfill(status=503, body="offline"))
        expect(page.locator("#rollout-connection")).to_contain_text("Cannot reach", timeout=10000)
        page.locator('[name="preset"][value="bad"]').check()
        expect(page.locator("#selected-model-name")).to_have_text("Broken labels")
        expect(page.locator("#start-rollout")).to_be_disabled()
        page.unroute("**/rollout")
        expect(page.locator("#start-rollout")).to_be_enabled(timeout=10000)
        page.locator("#show-playground").click()
        page.locator("#file-input").set_input_files("evaluation/images/coffee.png")
        page.locator("#show-rollout").click()
        page.locator(".bad-model .model-name").click()
        expect(page.locator("#start-rollout")).to_contain_text("broken labels")
        page.locator("#start-rollout").click()
        expect(page.locator("#traffic-caption")).to_contain_text("Candidate 10%")
        expect(page.locator("#rollout-result-title")).to_contain_text(
            "Automatic rollback", timeout=15000
        )
        expect(page.locator("#traffic-caption")).to_have_text("Stable 100% · Candidate 0%")
        expect(page.locator(".gate-fail")).to_have_count(1)
        page.wait_for_function(
            "document.getElementById('candidate-traffic').getBoundingClientRect().width < 1"
        )
        page.screenshot(path=str(output / "rollback.png"), full_page=True, animations="disabled")
        page.locator(".good-model .model-name").click()
        page.locator("#start-rollout").click()
        page.locator("#show-playground").click()
        expect(page.locator("#rollout-view")).to_be_hidden()
        expect(page.locator("#file-name")).to_have_text("coffee.png")
        page.locator("#show-rollout").click()
        expect(page.locator("#traffic-caption")).to_contain_text("Candidate 50%", timeout=15000)
        page.screenshot(
            path=str(output / "rollout-progress.png"), full_page=True, animations="disabled"
        )
        expect(page.locator("#rollout-result-title")).to_contain_text(
            "Rollout successful", timeout=20000
        )
        expect(page.locator(".gate-pass")).to_have_count(3)
        expect(page.locator("#current-model-name")).to_have_text("Reliable release")
        page.wait_for_function(
            "document.getElementById('candidate-traffic').getBoundingClientRect().width < 1"
        )
        page.screenshot(path=str(output / "rollout.png"), full_page=True, animations="disabled")

        # Invalid custom weights leave the promoted model in place and allow retry.
        page.locator('[name="model-source"][value="upload"]').check()
        expect(page.locator("#start-rollout")).to_be_disabled()
        expect(page.locator("#model-upload")).to_be_visible()
        page.screenshot(
            path=str(output / "rollout-upload.png"), full_page=True, animations="disabled"
        )
        with page.expect_file_chooser() as chooser:
            page.get_by_text("Choose model file", exact=True).click()
        chooser.value.set_files(
            {"name": "invalid.pth", "mimeType": "application/octet-stream", "buffer": b"bad"}
        )
        page.locator("#start-rollout").click()
        expect(page.locator("#rollout-error")).to_contain_text("Could not load model")
        expect(page.locator("#current-model-name")).to_have_text("Reliable release")
        expect(page.locator("#start-rollout")).to_be_enabled()

        # A real custom upload continues running across a page reload.
        page.locator("#model-file").set_input_files(model_path())
        page.locator("#start-rollout").click()
        expect(page.locator("#traffic-caption")).to_contain_text("Candidate 10%")
        page.reload()
        expect(page.locator("#rollout-result-title")).to_contain_text(
            "Rollout successful", timeout=20000
        )
        expect(page.locator("#current-model-name")).to_have_text("Your uploaded model")
        expect(page.locator(".gate-pass")).to_have_count(3)
        page.locator('[name="model-source"][value="preset"]').check()
        page.set_viewport_size({"width": 390, "height": 844})
        expect(page.locator("#rollout-view")).to_be_visible()
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        page.screenshot(
            path=str(output / "rollout-mobile.png"), full_page=True, animations="disabled"
        )
        page.locator("#show-playground").click()
        expect(page.locator("#rollout-view")).to_be_hidden()
        expect(page.locator("#show-playground")).to_have_attribute("aria-current", "page")
        page.go_back()
        expect(page.locator("#rollout-view")).to_be_visible()
        page.go_forward()
        expect(page.locator("#playground-view")).to_be_visible()
        page.locator("#file-input").set_input_files("evaluation/images/coffee.png")

        page.set_viewport_size({"width": 390, "height": 844})
        page.locator("#run-inference").click()
        expect(page.locator("#prediction-content")).to_be_visible()
        expect(page.locator("#run-inference")).to_be_enabled()
        expect(page.locator(".prediction")).to_have_count(3)
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        page.screenshot(path=str(output / "mobile.png"), full_page=True, animations="disabled")
        browser.close()
    assert not errors, errors
    print(f"Browser workflow passed. Screenshots: {output}")


if __name__ == "__main__":
    main()
