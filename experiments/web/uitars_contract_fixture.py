"""Execute synthetic native coordinates in a real browser, without a model."""
import argparse
import json
from pathlib import Path

from uitars_contract import file_sha256, parsed_action, screen_box


def run(contract_file, output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    contract = json.loads(Path(contract_file).read_text())
    from playwright.sync_api import sync_playwright
    width, height = contract['screenshot_size']
    mw, mh = contract['model_frame_size']
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=['--disable-gpu'])
        try:
            page = browser.new_page(viewport={'width': width, 'height': height}, device_scale_factor=1)
            page.set_content('<html><body><input id="proof" value="not ready"><button id="target" '
                'style="position:absolute;left:' + str(width / 2 - 20) + 'px;top:' + str(height / 2 - 20) +
                'px;width:40px;height:40px" onclick="document.querySelector(\'#proof\').value=\'ready\'">go</button></body></html>')
            text = "Thought: synthetic click.\nAction: click(point='<point>" + str(mw // 2) + ' ' + str(mh // 2) + "</point>')"
            action = parsed_action(text, contract)[0]
            point = screen_box(action['action_inputs']['start_box'], [width, height])
            page.mouse.click(*point)
            clicked = page.input_value('#proof') == 'ready'
            outside = "Thought: preserve a bad coordinate.\nAction: click(start_box='(" + str(mw + 28) + ',' + str(mh // 2) + ")')"
            outside_action = parsed_action(outside, contract)[0]
            outside_point = screen_box(outside_action['action_inputs']['start_box'], [width, height])
            preserved_error = outside_point[0] > width
            page.screenshot(path=str(out / 'synthetic_click.png'))
            result = {'kind': 'scripted native-parser/browser fixture; excluded from all benchmark denominators',
                'fixture_passed': clicked and preserved_error, 'actual_model_requests': 0, 'benchmark_scores': 0,
                'frame_contract_sha256': file_sha256(contract_file), 'scripted_screen_point': point,
                'retained_DOM_state': page.input_value('#proof'), 'out_of_frame_prediction_not_clamped': preserved_error,
                'out_of_frame_screen_point': outside_point}
            (out / 'preflight.json').write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(result))
            if not result['fixture_passed']:
                raise RuntimeError('Native coordinate and browser fixture failed')
        finally:
            browser.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.contract, args.output)
