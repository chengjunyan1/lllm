"""
Example script that loads the bundled proxies and prints a short catalog.

Run with:

    python examples/proxy_catalog.py
"""

from lllm.proxies import Proxy, load_builtin_proxies


def main():
    loaded, errors = load_builtin_proxies()
    if loaded:
        print("Loaded proxy modules:")
        for module in loaded:
            print(f"  - {module}")
    if errors:
        print("\nSome proxies could not be imported (missing optional deps?):")
        for module, exc in errors.items():
            print(f"  - {module}: {exc}")

    proxy = Proxy()
    available = proxy.available()
    if not available:
        print("\nNo proxies are currently registered.")
        return

    print("\nAvailable proxy identifiers:")
    for name in available:
        print(f"  - {name}")

    sample = available[0]
    print(
        f"\nUse Proxy like: proxy('{sample}.your_endpoint', params=...) once you "
        "link functions to prompts."
    )


if __name__ == "__main__":
    main()
