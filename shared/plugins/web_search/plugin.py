"""Web search plugin for performing internet searches."""

from typing import Dict, List, Any, Callable, Optional

from ..base import UserCommand
from ..model_provider.types import ToolSchema


DEFAULT_MAX_RESULTS = 10
DEFAULT_TIMEOUT = 10  # seconds


class WebSearchPlugin:
    """Plugin that provides web search capability using DuckDuckGo.

    Configuration:
        max_results: Maximum number of search results to return (default: 10).
        timeout: Request timeout in seconds (default: 10).
        region: Region for search results (default: 'wt-wt' for no region).
        safesearch: Safe search level - 'off', 'moderate', 'strict' (default: 'moderate').
    """

    def __init__(self):
        self._max_results: int = DEFAULT_MAX_RESULTS
        self._timeout: int = DEFAULT_TIMEOUT
        self._region: str = 'wt-wt'  # No specific region
        self._safesearch: str = 'moderate'
        self._initialized = False
        self._ddgs = None

    @property
    def name(self) -> str:
        return "web_search"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the web search plugin.

        Args:
            config: Optional dict with:
                - max_results: Maximum number of results to return (default: 10)
                - timeout: Request timeout in seconds (default: 10)
                - region: Region code for results (default: 'wt-wt')
                - safesearch: Safe search level (default: 'moderate')
        """
        if config:
            if 'max_results' in config:
                self._max_results = config['max_results']
            if 'timeout' in config:
                self._timeout = config['timeout']
            if 'region' in config:
                self._region = config['region']
            if 'safesearch' in config:
                self._safesearch = config['safesearch']
        self._initialized = True

    def shutdown(self) -> None:
        """Shutdown the web search plugin."""
        self._ddgs = None
        self._initialized = False

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return the ToolSchema for the web search tool."""
        return [ToolSchema(
            name='web_search',
            description='Search the web for information on any topic',
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find information about"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 10)"
                    }
                },
                "required": ["query"]
            }
        )]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return the executor mapping."""
        return {'web_search': self._execute}

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions for the web search tool."""
        return """You have access to `web_search` which searches the web for current information.

Use it to find up-to-date information about any topic, including:
- Current events and news
- Technical documentation and tutorials
- Product information and reviews
- Research and academic topics
- Any information that may have changed since your training cutoff

Example usage:
- Search for news: web_search(query="latest AI developments 2024")
- Find documentation: web_search(query="Python asyncio tutorial")
- Look up information: web_search(query="climate change statistics 2024")

The tool returns a list of search results with titles, URLs, and snippets.

Tips for effective searches:
- Be specific with your queries for better results
- Include relevant keywords and context
- Use quotes for exact phrase matching
- Add year/date for time-sensitive information"""

    def get_auto_approved_tools(self) -> List[str]:
        """Web search is read-only and safe - auto-approve it."""
        return ['web_search']

    def get_user_commands(self) -> List[UserCommand]:
        """Web search plugin provides model tools only, no user commands."""
        return []

    def _execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a web search.

        Args:
            args: Dict containing:
                - query: The search query (required)
                - max_results: Optional max results override

        Returns:
            Dict containing search results or error.
        """
        try:
            query = args.get('query')
            if not query:
                return {'error': 'web_search: query must be provided'}

            max_results = args.get('max_results', self._max_results)

            # Lazy import to avoid startup cost if plugin not used
            try:
                from ddgs import DDGS
            except ImportError:
                return {
                    'error': 'ddgs package not installed',
                    'hint': 'Install with: pip install ddgs'
                }

            # Perform the search
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    query,
                    region=self._region,
                    safesearch=self._safesearch,
                    max_results=max_results
                ))

            if not results:
                return {
                    'query': query,
                    'results': [],
                    'message': 'No results found for the query'
                }

            # Format results for the model
            formatted_results = []
            for r in results:
                formatted_results.append({
                    'title': r.get('title', ''),
                    'url': r.get('href', r.get('link', '')),
                    'snippet': r.get('body', r.get('snippet', ''))
                })

            return {
                'query': query,
                'result_count': len(formatted_results),
                'results': formatted_results
            }

        except Exception as exc:
            return {'error': f'web_search failed: {str(exc)}'}


def create_plugin() -> WebSearchPlugin:
    """Factory function to create the web search plugin instance."""
    return WebSearchPlugin()
