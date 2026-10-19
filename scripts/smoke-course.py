"""Check course styling, outline navigation and live experiment controls."""
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1000})
    page.set_default_timeout(60000)
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:8000')
    assert page.locator('.course-sidebar').is_visible()
    assert page.locator('#course-progress').get_attribute('value')=='0'
    page.screenshot(path='artifacts/course-desktop.png')
    page.locator('.outline-link[href="#predict"]').click()
    page.locator('#q-prediction-form button').click()
    page.wait_for_function("document.querySelector('#q-prediction-status').textContent.includes('Real forward pass')")
    page.locator('#q-greedy').click()
    page.wait_for_function("!document.querySelector('#q-greedy').disabled")
    assert page.locator('#q-current-text').inner_text()!='Text so far: The sky is'
    page.locator('.finish-step[data-step="3"]').click()
    page.wait_for_function("document.querySelector('#course-progress').value===1")
    assert page.locator('[data-outline-step="3"]').get_attribute('class').find('is-complete')>=0
    page.locator('#predict').scroll_into_view_if_needed()
    page.screenshot(path='artifacts/course-experiment.png')
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    page.set_viewport_size({'width':390,'height':844})
    page.goto('http://127.0.0.1:8000')
    assert not page.locator('.course-outline').evaluate('(n)=>n.open')
    page.screenshot(path='artifacts/course-mobile.png')
    page.locator('.course-outline>summary').click()
    page.locator('.outline-link[href="#predict"]').click()
    assert not page.locator('.course-outline').evaluate('(n)=>n.open')
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    assert not errors,errors
    print('Desktop/mobile layout, outline links, completion progress, and real Qwen generation passed.')
    browser.close()
