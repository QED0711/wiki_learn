import requests
from bs4 import BeautifulSoup

from url_utils import parse_url, get_title


class WikiIntroScrapper:
    
    def __init__(self, url):
        self.url = url
        
        self.title = None
        
        self.intro_links = []
        self.parsed_links = []
        self.intro_link_titles = []
    
    def _get_article(self):
        resp = requests.get(self.url)
        soup = BeautifulSoup(resp.content)
        self._set_title(soup)
        return soup
        
    def _set_title(self, soup):
        self.title = soup.find_all("h1")[0].text

    
    def _get_intro_links(self):
        soup = self._get_article().find_all(class_="mw-parser-output")[0]
        
        children = list(soup.children)
        
        started_intro_text = False
        
        for child in children:
            if child.name == "p" and child.has_attr("class") == False:
                started_intro_text = True
                self.intro_links += child.find_all("a")
            if child.name != "p" and started_intro_text:
                break
    
    def parse_intro_links(self):
        self._get_intro_links()
        
        for link in self.intro_links:
            current_href = link.get('href')
            if current_href.startswith("/wiki/") and not (":" in current_href):
                self.parsed_links.append(parse_url("https://en.wikipedia.org" + current_href))
                self.intro_link_titles.append(get_title(current_href))

        
        return self.parsed_links


if __name__ == "__main__":
    ws = WikiIntroScrapper("https://en.wikipedia.org/wiki/Random_forest")

    ws.parse_intro_links()

    print("Title:\t", ws.title)

    print(ws.intro_link_titles)