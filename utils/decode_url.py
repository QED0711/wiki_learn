from urllib.parse import unquote

def parse_url(url):
    return unquote(url)

if __name__ == "__main__":
    print(parse_url("https://en.wikipedia.org/wiki/Elisabeth_R%C3%B6ckel"))