from langchain_core.tools import tool
from langchain_community.tools import TavilySearchResults


class WeatherTool:
    """A tool to get the weather information for a given location using a web-based search"""

    @tool
    @staticmethod
    def get_weather(location: str):
        """Get the weather information based on the user's query."""
        search_tool = TavilySearchResults(
            max_results=3,
            include_answer=True,
            include_raw_content=False,
            include_images=False,
            search_depth="advanced",
        )

        query = f"weather information for {location}"
        results = search_tool.invoke({"query": query})
        if not results:
            return "Unable to find weather information for {location}"
        
        # Build a snippet of info from the results
        snippet = ""
        for result in results:
            snippet += result.get("content","")[:300] + "\n"
        
        return snippet
    