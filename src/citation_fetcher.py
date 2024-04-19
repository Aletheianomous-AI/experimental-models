from duckduckgo_search import DDGS as searcher
from duckduckgo_search.exceptions import DuckDuckGoSearchException as ddgse

class Citation_Fetcher():
    """This class searches the query online."""

    def parse_results(raw_results: dict):
        """This function parses the results from the json file
            as return by DDGS.

            PARAMETERS
            raw_results - The search results returned from DDGS
                as a dict.
        """

        results_output = ""
        i = 1
        for item in raw_results:
            results_output += "["  + str(i) + "] "
            results_output += item['href'] + " : "
            results_output += item['body'] + "\n"
            i+=1
        return results_output
    
    def search_online(query):
        """This function searches the internet online from
            the search query.        
        """

        raw_results = None
        try:
            search_instance = searcher()
            raw_results = search_instance.text(query, max_results=3)
            results = Citation_Fetcher.parse_results(raw_results)
        except ddgse:
            raise ddgse()
        return raw_results, results
