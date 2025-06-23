import asyncio
import logging
import cloudscraper
from bs4 import BeautifulSoup
import aioodbc
from urllib.parse import urljoin, unquote
from typing import List, Tuple, Optional
import re
import json
from datetime import datetime
import random
from fake_useragent import UserAgent
from seleniumbase import SB
import time

# Configuration
BATCH_SIZE = 5
REQUEST_DELAY = 2.5  # Increased delay to avoid detection
MIN_DELAY = 1.5  # Minimum delay between requests in seconds
MAX_DELAY = 5.0  # Maximum delay between requests
START_CVR = 10000024
MAX_RETRIES = 3
EMPTY_BATCH_SLEEP = 60  # Seconds to wait when no companies found
BETWEEN_BATCH_DELAY = (20, 40)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'scraper_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)


class WebsiteScraper:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ua = UserAgent()
        self.scraper = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'windows',
                'mobile': False
            },
            delay=REQUEST_DELAY
        )

        self.base_headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        }

    def _get_random_headers(self):
        headers = self.base_headers.copy()
        headers['User-Agent'] = self.ua.random
        return headers

    async def _make_request(self, url: str, method: str = 'GET', timeout: int = 30) -> Optional[str]:
        for attempt in range(MAX_RETRIES):
            try:
                headers = self._get_random_headers()
                response = self.scraper.request(
                    method=method,
                    url=url,
                    headers=headers,
                    timeout=timeout
                )
                self.logger.info(f"{response.status_code}")
                if response.status_code == 403:
                    content = response.text.lower()
                    verification_indicators = [
                        'verify you are human',
                        'human verification',
                        'cloudflare',
                        'turnstile',
                        'captcha',
                        'bekræft, at du er menneske'
                    ]

                    if any(indicator in content for indicator in verification_indicators):
                        self.logger.warning(f"Cloudflare detected on {url}. Switching to Selenium...")
                        await asyncio.sleep(random.uniform(40.0, 60.0))
                        return await self._selenium_request(url)

                return response.text if response.status_code == 200 else None

            except Exception as e:
                self.logger.error(f"Request error for {url} (attempt {attempt+1}/{MAX_RETRIES}): {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
        return None

    async def _selenium_request(self, url: str, retries: int = MAX_RETRIES) -> Optional[str]:
        verification_indicators = [
            'verify you are human',
            'human verification',
            'cloudflare',
            'turnstile',
            'captcha',
            'bekræft, at du er menneske'
        ]
        for attempt in range(retries):
            try:
                with SB(uc=True, test=True, locale_code="en") as sb:
                    sb.driver.uc_open_with_reconnect(url, 1)
                    time.sleep(5)

                    if any(indicator in sb.driver.page_source.lower() for indicator in verification_indicators):
                        self.logger.info(f"Cloudflare detected on {url}. Handling challenge...")
                        sb.driver.reconnect(0.1)
                        sb.driver.uc_open_with_reconnect(url, 1)
                        time.sleep(5)
                        sb.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                        time.sleep(5)
                        return sb.get_page_source()
                    else:
                        return self.seleniumbasescrape_degulsider_dk(url)

            except Exception as e:
                self.logger.error(f"Selenium error for {url}: {str(e)}")
                time.sleep(random.uniform(5.0, 10.0))
        return None

    @staticmethod
    def clean_address(address: str) -> str:
        """Clean and normalize addresses."""
        if not address:
            return ""

        address = address.strip()
        if "Postboks" in address:
            match = re.search(r"Postboks \d+ (\d{4} \w+(?:\s+[A-ZÆØÅ])?)", address)
            return match.group(1) if match else address

        matches = re.findall(r"(\d{4} \w+(?:\s+[A-ZÆØÅ])?)", address)
        return matches[-1] if matches else address

    def clean_url(self, url: str) -> str:
        """Clean and normalize URLs."""
        if not url:
            return ""

        url = unquote(url)
        url = re.sub(r'^https?://(www\.)?', '', url.lower())
        url = url.rstrip('/')
        return url

    async def scrape_krak_dk(self, company_name: str, address: str, cvr_number: int) -> Optional[str]:
        """Scrape Krak.dk with improved anti-bot handling."""
        try:
            cleaned_address = self.clean_address(address)
            formatted_search = f"{company_name} {cleaned_address}".replace(' ', '+').replace('/', '')

            self.logger.info(f"Searching Krak.dk for: {formatted_search}")

            # Add random delay
            await asyncio.sleep(random.uniform(1.0, 3.0))

            search_url = f"https://www.krak.dk/{formatted_search}/firmaer"

            html_content = await self._make_request(search_url)
            if not html_content:
                self.logger.info(f"No html_content found with regular request on Krak.dk for CVR {cvr_number}, trying SeleniumBase")
                return self.seleniumbasescrape_degulsider_dk(company_name, address, cvr_number)

            soup = BeautifulSoup(html_content, 'html.parser')
            return await self._extract_website_from_soup(soup, cvr_number, site='krak')

        except Exception as e:
            self.logger.error(f"Krak.dk error for CVR {cvr_number}: {str(e)}")
            return None

    async def scrape_118_dk(self, cvr: int) -> Optional[str]:
        """Scrape 118.dk with improved error handling and retries."""
        try:
            self.logger.info(f"Searching 118.dk for CVR: {cvr}")

            search_url = f'https://118.dk/advanced/go?pageSize=25&advanced=1&whatFirstName=&whatLastName=&whatBusinessName=&whatCategory=&whatPhone=&whereStreet=&whereHouseNumber=&whereZip=&whereCity=&whatOther={cvr}'
            html_content = await self._make_request(search_url)
            if not html_content:
                return None

            soup = BeautifulSoup(html_content, 'html.parser')
            detail_link = soup.find('a', class_='btn-se-kort')
            if not detail_link:
                return None

            detail_url = urljoin('https://118.dk/', detail_link.get('href', ''))
            detail_html_content = await self._make_request(detail_url)
            if not detail_html_content:
                return None

            detail_soup = BeautifulSoup(detail_html_content, 'html.parser')
            website_element = detail_soup.find('a', {'data-tracking-label': 'business_website'})

            if website_element and website_element.get('href'):
                website = self.clean_url(website_element['href'])
                self.logger.info(f"Found website on 118.dk for CVR {cvr}: {website}")
                return website

            return None

        except Exception as e:
            self.logger.error(f"118.dk error for CVR {cvr}: {str(e)}")
            return None


    def seleniumbasescrape_degulsider_dk(self, company_name: str, address: str, cvr_number: int) -> Optional[str]:
        """Scrape Krak.dk with Cloudflare bypass."""
        with SB(uc=True) as sb:
            try:
                cleaned_address = self.clean_address(address)
                formatted_search = f"{company_name} {cleaned_address}".replace(' ', '+').replace('/', '')
                self.logger.info(f"Searching Degulesider.dk for: {formatted_search}")

                try:
                    sb.open(f"https://www.degulesider.dk/{formatted_search}/firmaer")
                    time.sleep(random.uniform(1.0, 2.0))
                    sb.wait_for_ready_state_complete()
                except Exception as e:
                    logging.error(f"Page navigation failed: {e}")
                    return None

                    # Wait for the search results to load
                sb.wait_for_element("span.text-small.text-neutral-200", timeout=5)

                # Extract JSON-LD data
                json_ld_script = sb.execute_script("""
                                const scripts = Array.from(document.querySelectorAll('script[type="application/ld+json"]'));
                                return scripts.length ? scripts[0].textContent : null;
                            """)
                json_ld = json.loads(json_ld_script) if json_ld_script else None

                if not json_ld or not json_ld.get('itemListElement'):
                    logging.warning(f"No companies found for: {company_name}")
                    return None

                for company in json_ld['itemListElement']:
                    detail_url = company.get('url')
                    if not detail_url:
                        continue

                    try:
                        sb.open(detail_url)
                        sb.wait_for_ready_state_complete()
                        time.sleep(1)  # Let page stabilize after load

                        # First check if cookie banner exists before waiting
                        if sb.is_element_present("#CybotCookiebotDialog"):
                            try:
                                # Wait and click in a single operation to avoid stale element
                                sb.click_if_visible("#CybotCookiebotDialogBodyButtonDecline", timeout=5)
                                logging.info("Cookie banner dismissed successfully")

                                # Wait for banner to disappear before proceeding
                                sb.wait_for_element_not_visible("#CybotCookiebotDialog", timeout=5)
                            except Exception as e:
                                logging.info(f"Cookie banner handling failed: {e}")

                        # Ensure page is stable after cookie handling
                        sb.wait_for_ready_state_complete()

                        # Now proceed with CVR element search
                        cvr_element = sb.wait_for_element(
                            "//p[contains(., 'CVR-nr:')]/following-sibling::p",
                            timeout=20
                        )
                        cvr_number = cvr_element.text
                        extracted_cvr = None

                        if cvr_element:
                            extracted_cvr = cvr_element.text.strip()

                        if extracted_cvr != str(cvr_number):
                            continue

                        # Wait for website links to appear
                        website_links = sb.find_elements('a[data-guv-click="company_website_link"]')
                        for link in website_links:
                            href = link.get_attribute('href')
                            if href:
                                website = href.split('url=')[1].split('&')[
                                    0] if 'degulesider.dk/exit?url=' in href else href
                                print(f"Found website for CVR {cvr_number}: {website}")
                                return website



                    except Exception as detail_err:
                        logging.warning(f"Detail page error: {detail_err}")
                        continue

                logging.info(f"No website found for CVR {cvr_number}")
                return None
            except Exception as e:
                logging.error(f"degulesider error: {e}")
                return None


    async def _extract_website_from_soup(self, soup: BeautifulSoup, cvr_number: int, site: str = 'krak') -> Optional[str]:
        """Extract website from BeautifulSoup object."""
        self.logger.info(f"Extracting website from {site} for CVR {cvr_number}")
        try:
            self.logger.debug(f"Soup content: {soup.prettify()}")
            json_ld_script = soup.find('script', {'type': 'application/ld+json'})
            if not json_ld_script:
                self.logger.warning(f"No JSON-LD script found on {site}")
                return None

            try:
                json_ld = json.loads(json_ld_script.string)
            except json.JSONDecodeError as e:
                self.logger.error(f"Error parsing {site} JSON-LD: {str(e)}")
                return None

            if not json_ld.get('itemListElement'):
                self.logger.warning(f"No companies found on {site}")
                return None

            for company in json_ld['itemListElement']:
                detail_url = company.get('url')
                if not detail_url:
                    continue

                self.logger.info(f"Checking {site} detail page: {detail_url}")
                detail_html_content = await self._make_request(detail_url)
                if not detail_html_content:
                    continue

                detail_soup = BeautifulSoup(detail_html_content, 'html.parser')
                cvr_element = detail_soup.find('p', string=re.compile(r'CVR-nr:'))
                if not cvr_element:
                    continue

                extracted_cvr = cvr_element.find_next_sibling('p').text.strip()
                if extracted_cvr != str(cvr_number):
                    self.logger.info(
                        f"CVR mismatch on {site} - Expected: {cvr_number}, Found: {extracted_cvr}")
                    continue

                website_element = detail_soup.find('a', {'data-guv-click': 'company_website_link'})
                if website_element and website_element.get('href'):
                    href = website_element['href']
                    website = href.split('url=')[1].split('&')[0] if f'{site}.dk/exit?url=' in href else href
                    cleaned_website = self.clean_url(website)
                    self.logger.info(f"Found website on {site} for CVR {cvr_number}: {cleaned_website}")
                    return cleaned_website

            return None
        except Exception as e:
            self.logger.error(f"Error extracting website from {site}: {str(e)}")
            return None

    @staticmethod
    def _verify_cvr_match(item: dict, cvr_number: int) -> bool:
        """Verify if the item matches the CVR number."""
        return str(item.get('cvr', '')).strip() == str(cvr_number)

    def _extract_website_url(self, item: dict) -> Optional[str]:
        """Extract and clean website URL from item."""
        website = item.get('url')
        if website:
            return self.clean_url(website)
        return None


async def scrape_company(company: Tuple[int, str, str], scraper: WebsiteScraper) -> Tuple[str, int]:
    cvr, name, address = company
    logging.info(f"Processing: {name} (CVR: {cvr})")  # Fixed: removed self.logger

    try:
        # Try Krak.dk first
        website = await scraper.scrape_krak_dk(name, address, cvr)

        # Fallback to 118.dk
        if not website:
            logging.info(f"No website found on Krak.dk for CVR {cvr}, trying 118.dk")
            website = await scraper.scrape_118_dk(cvr)

        return (website or "no website", cvr)

    except Exception as e:
        logging.error(f"Error processing company {name} (CVR: {cvr}): {str(e)}")
        return ("no website", cvr)


async def get_db_pool() -> aioodbc.Pool:
    try:
        return await aioodbc.create_pool(
            dsn="DRIVER={SQL ServerServer};SERVER=DESKTOP-VPRC2OF;DATABASE=cvr_database;Trusted_Connection=yes;",
            autocommit=False,
            minsize=1,
            maxsize=10,
            timeout=10
        )
    except Exception as e:
        logging.critical(f"Database connection failed: {str(e)}")
        raise


async def fetch_companies(pool: aioodbc.Pool, batch_size: int, last_cvr: int) -> List[Tuple[int, str, str]]:
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute("""
                SELECT TOP (?) cvr_number, name, address
                FROM companies
                WHERE cvr_number > ?
                AND (defaultwebpage IS NULL)
                ORDER BY cvr_number
            """, (batch_size, last_cvr))
            return await cursor.fetchall()


async def process_batch(batch: List[Tuple[int, str, str]], pool: aioodbc.Pool) -> None:
    try:
        scraper = WebsiteScraper()
        results = []
        for company in batch:
            start_time = time.time()

            try:
                result = await scrape_company(company, scraper)
                results.append(result)
            except Exception as e:
                logging.error(f"Failed processing CVR {company[0]}: {str(e)}")
                results.append(("no website", company[0]))

            # Randomized delay between requests
            elapsed = time.time() - start_time
            remaining_delay = max(0, random.uniform(MIN_DELAY, MAX_DELAY) - elapsed)
            await asyncio.sleep(remaining_delay)

        # Update database
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.executemany(
                    "UPDATE companies SET defaultwebpage = ? WHERE cvr_number = ?",
                    results
                )
                await conn.commit()

    except Exception as e:
        logging.error(f"Batch processing failed: {str(e)}")


async def process_companies() -> None:
    pool = None
    last_cvr = START_CVR
    processed_total = 0

    try:
        pool = await get_db_pool()

        while True:
            batch = await fetch_companies(pool, BATCH_SIZE, last_cvr)

            if not batch:
                logging.info(f"No companies found. Retrying in {EMPTY_BATCH_SLEEP}s...")
                await asyncio.sleep(EMPTY_BATCH_SLEEP)
                continue

            start_time = datetime.now()
            await process_batch(batch, pool)

            # Update tracking
            last_cvr = max(cvr for (cvr, _, _) in batch)
            processed_total += len(batch)

            # Random delay between batches
            batch_delay = random.uniform(*BETWEEN_BATCH_DELAY)
            logging.info(f"Waiting {batch_delay:.2f}s before next batch...")
            await asyncio.sleep(batch_delay)

    except asyncio.CancelledError:
        logging.info("Shutdown initiated...")
    finally:
        if pool:
            pool.close()
            await pool.wait_closed()


if __name__ == '__main__':
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        task = loop.create_task(process_companies())
        loop.run_forever()
    except KeyboardInterrupt:
        logging.info("\nGraceful shutdown requested...")
        task.cancel()
        loop.run_until_complete(task)
    finally:
        loop.close()