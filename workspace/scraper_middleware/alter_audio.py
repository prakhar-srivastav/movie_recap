

from playwright.sync_api import sync_playwright

def run(playwright, file_path, affects):
    browser = playwright.chromium.launch(headless=True)  # Set headless=False to see the browser
    page = browser.new_page()

    input_f = '#__next > div > div > form > div > input[type=file]' # Input a file
    click_s = '#__next > div > div > form > div > div > div.css-0 > button' # click submit
    click_d = "#__next > div > div > div.css-8fg0jl > ul > li > div > a" # download

    url = 'https://audioalter.com/preset/{}'.format(affects)
    page.goto(url)
    page.set_input_files(input_f, file_path)
    page.wait_for_load_state()
    page.wait_for_timeout(3000)

    page.locator(click_s).click()
    page.wait_for_load_state()
    page.wait_for_timeout(3000)

    with page.expect_download() as download_info:
        page.locator(click_d).click()
        download = download_info.value
        download.save_as(file_path)
    browser.close()


def alter_audio(file_path, affects = 'slowed-reverb'):
    suff = file_path.split('.')[-1]
    import os
    old_file_path = file_path
    new_file_path = file_path.replace(suff,'mp3')

    if old_file_path != new_file_path:
        os.rename(old_file_path, new_file_path)

    with sync_playwright() as playwright:
        run(playwright, new_file_path, affects)

    if old_file_path != new_file_path:
        os.rename(new_file_path, old_file_path)