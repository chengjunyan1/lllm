#!/usr/bin/env python3
"""
Sample MCP Server for Deep Research API Integration

This server implements the Model Context Protocol (MCP) with search and fetch
capabilities designed to work with ChatGPT's deep research feature.

Run this script to start the MCP server.
Then run ngrok to expose the server to the internet. e.g.:

ngrok http --url=jserver.ngrok.io 8000
"""
import datetime as dt
import os
from exa_py import Exa
import logging
from typing import Dict, List, Any
from fastmcp import FastMCP
import requests
from fake_useragent import UserAgent
import concurrent.futures
from googlesearch import search

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXA_API_KEY = os.getenv("EXA_API_KEY")
exa_client = Exa(EXA_API_KEY)


CUTOFF_DATE = '2024-06-01'
NUM_RESULTS = 25 # note: if INCLUDE_TEXT is True, then for every page, it will fetch the full text, may cause the price high
EXA_SEARCH_CUTOFF_DATE = dt.datetime.strptime(CUTOFF_DATE, '%Y-%m-%d')
EXA_SEARCH_MAX_LENGTH = 200


USE_EXA_SEARCH = False # if True, will use exa search, otherwise will use google search
USE_EXA_SEARCH_IF_FAIL = True # if True, will use exa search if google search fails
USE_EXA_FETCH = False
EXA_SEARCH_TYPE = 'keyword' # 'auto', 'neural', 'keyword'
INCLUDE_TEXT_EXA = False
FETCH_FULL_TEXT_SEARCH = True # WILL BE IGNORED IF INCLUDE_TEXT_EXA IS TRUE



PORT = 8000



def search_google(query, end_date, num_results):
    if end_date:
        query += f" until:{end_date}"
    _results = search(query, num_results=num_results, unique=True, advanced=True)
    results = [{
        'id': i.url,
        'title': i.title,
        'text': i.description,
        'url': i.url
    } for i in _results]
    return results


ua = UserAgent()

def get_web_content(url): # OPTIONAL: cheaper alternative to exa_client.get_contents
    """
    Fetches the content of a given URL.

    Args:
        url (str): The URL of the webpage to fetch.

    Returns:
        str: The content of the webpage as a string, or None if an error occurs.
    """
    try:
        user_agent = ua.random
        headers = {
            "User-Agent": user_agent
        }
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.text
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {e}"

def get_web_contents(urls):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(get_web_content, urls))
    return results


def exa_search(query):
    args = {
        'end_published_date': EXA_SEARCH_CUTOFF_DATE.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
        'type': EXA_SEARCH_TYPE,
    }
    if EXA_SEARCH_TYPE != 'keyword':
        args['end_crawl_date'] = EXA_SEARCH_CUTOFF_DATE.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

    if INCLUDE_TEXT_EXA:
        response = exa_client.search_and_contents(
            query,
            num_results=NUM_RESULTS,
            text=True,
            **args
        )
    else:
        response = exa_client.search(
            query,
            num_results=NUM_RESULTS,
            **args
        )

    results = []
    # Process the vector store search results
    if FETCH_FULL_TEXT_SEARCH and not INCLUDE_TEXT_EXA:
        urls = [r.url for r in response.results]
        text_contents = get_web_contents(urls)

    for idx, r in enumerate(response.results):
        if INCLUDE_TEXT_EXA:
            text_content = r.text
        elif FETCH_FULL_TEXT_SEARCH:
            text_content = text_contents[idx]
        else:   
            text_content = r.title
        # Create a snippet from content
        text_snippet = text_content[:EXA_SEARCH_MAX_LENGTH] + "..." if len(
            text_content) > EXA_SEARCH_MAX_LENGTH else text_content

        result = {
            "id": r.url,
            "title": r.title,
            "text": text_snippet,
            "url": r.url
        }
        results.append(result)
    return results


def create_server():
    """Create and configure the MCP server with search and fetch tools."""

    # Initialize the FastMCP server
    mcp = FastMCP(name="Search MCP Server",
                  instructions="""
        This MCP server provides web search and web content retrieval capabilities.
        Use the search tool to find relevant web pages based on keywords, then use the fetch 
        tool to retrieve complete web page content.
        """)

    @mcp.tool()
    async def search(query: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search for web pages.
        
        This tool searches through the web to find semantically relevant web pages.
        Returns a list of search results with basic information. Use the fetch tool to get 
        complete web page content.
        
        Args:
            query: Search query string. Keyword based search works best.
        
        Returns:
            Dictionary with 'results' key containing list of matching web pages.
            Each result includes id, title, text snippet, and optional URL.
        """
        if not query or not query.strip():
            return {"results": []}


        if USE_EXA_SEARCH:
            logger.info(
                f"Searching exa for query: '{query}'")
            results = exa_search(query)
        else:
            try:
                logger.info(f"Searching google for query: '{query}'")
                results = search_google(query, CUTOFF_DATE, NUM_RESULTS)
            except Exception as e:
                if USE_EXA_SEARCH_IF_FAIL:
                    logger.info(f"Google search failed, searching exa for query: '{query}'")
                    results = exa_search(query)
                else:
                    raise e

        logger.info(f"Search returned {len(results)} results")
        return {"results": results}

    @mcp.tool()
    async def fetch(id: str) -> Dict[str, Any]:
        """
        Retrieve complete web page content by URL for detailed analysis and citation.
        
        This tool fetches the full web page content.
        Use this after finding relevant web pages with the search tool to get complete 
        information for analysis and proper citation.
        
        Args:
            id: Web page id (URL of the web page to fetch)
            
        Returns:
            Complete web page with id, title, full text content, optional URL, and metadata
            
        Raises:
            ValueError: If the specified URL is not found
        """
        if not id:
            raise ValueError("ID is required")

        logger.info(f"Fetching content for ID: {id}")

        if USE_EXA_FETCH:
            response = exa_client.get_contents(
                urls=[id],
                text=True,
            )
            web_page = response.results[0]
            web_page_content = web_page.text
            web_page_title = web_page.title
        else:   
            web_page_title = id
            web_page_content = get_web_content(id)

        result = {
            "id": id,
            "title": web_page_title,
            "text": web_page_content,
            "url": id,
            "metadata": None
        }

        logger.info(f"Successfully fetched web page: {id}")
        return result

    return mcp


def main():
    """Main function to start the MCP server."""
    # Verify OpenAI client is initialized
    # Create the MCP server
    server = create_server()

    # Configure and start the server
    logger.info(f"Starting MCP server on 0.0.0.0:{PORT}")
    logger.info("Server will be accessible via SSE transport")
    logger.info("Connect this server to ChatGPT Deep Research for testing")

    try:
        # Use FastMCP's built-in run method with SSE transport
        server.run(transport="http", host="0.0.0.0", port=PORT)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    main()