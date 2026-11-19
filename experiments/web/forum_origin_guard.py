"""Keep legacy absolute Forum links inside the current owned environment."""
from contextlib import contextmanager
import hashlib
from urllib.parse import urlsplit, urlunsplit


def origins(original):
    # Official VWA's Forum URL mapping uses reddit.com in model prompts.
    return [original, 'http://reddit.com', 'https://reddit.com',
            'http://www.reddit.com', 'https://www.reddit.com']


def declared_origin(url, original):
    parsed = urlsplit(url)
    for origin in origins(original):
        expected = urlsplit(origin)
        if (parsed.scheme, parsed.netloc) == (expected.scheme, expected.netloc):
            return origin
    raise ValueError('URL is outside the declared Forum origin mappings')


def destination(url, original, current):
    parsed, before, after = urlsplit(url), urlsplit(original), urlsplit(current)
    if (parsed.scheme, parsed.netloc) != (before.scheme, before.netloc):
        raise ValueError('URL does not belong to the declared original Forum origin')
    if (after.scheme, after.netloc) == (before.scheme, before.netloc) or after.scheme not in ('http', 'https'):
        raise ValueError('An independent HTTP environment is required')
    return urlunsplit((after.scheme, after.netloc, parsed.path, parsed.query, parsed.fragment))


def note(audit, url, mode, status):
    audit.append({'original_url_sha256': hashlib.sha256(url.encode()).hexdigest(),
                  'status': status, 'mode': mode, 'method_and_body_preserved': True})


def cors_headers(current, requested=''):
    return {'access-control-allow-origin': current,
            'access-control-allow-credentials': 'true',
            'access-control-allow-methods': 'GET, HEAD, POST, PUT, PATCH, DELETE, OPTIONS',
            'access-control-allow-headers': requested}


async def guard_async_context(context, original, current, audit):
    async def redirect(route):
        url = route.request.url
        try:
            target = destination(url, declared_origin(url, original), current)
            if route.request.method == 'OPTIONS':
                note(audit, url, 'local_preflight', 204)
                await route.fulfill(status=204, headers=cors_headers(current, route.request.headers.get('access-control-request-headers', '')), body='')
            elif route.request.is_navigation_request():
                note(audit, url, 'navigation_redirect', 307)
                await route.fulfill(status=307, headers={'location': target}, body='')
            else:
                response = await route.fetch(url=target)
                note(audit, url, 'owned_subresource_proxy', response.status)
                await route.fulfill(response=response, headers={**response.headers, **cors_headers(current)})
                await response.dispose()
        except Exception as error:
            audit.append({'event': 'guard_error', 'error': type(error).__name__})
            await route.abort()
    for origin in origins(original):
        await context.route(origin.rstrip('/') + '/**', redirect)
    audit.append({'event': 'context_guard_installed', 'mode': 'async'})


@contextmanager
def guard_sync_contexts(original, current, audit):
    if not original:
        yield
        return
    from playwright.sync_api import Browser
    before = Browser.new_context
    def create(browser, *args, **kwargs):
        context = before(browser, *args, **kwargs)
        def redirect(route):
            url = route.request.url
            try:
                target = destination(url, declared_origin(url, original), current)
                if route.request.method == 'OPTIONS':
                    note(audit, url, 'local_preflight', 204)
                    route.fulfill(status=204, headers=cors_headers(current, route.request.headers.get('access-control-request-headers', '')), body='')
                elif route.request.is_navigation_request():
                    note(audit, url, 'navigation_redirect', 307)
                    route.fulfill(status=307, headers={'location': target}, body='')
                else:
                    response = route.fetch(url=target)
                    note(audit, url, 'owned_subresource_proxy', response.status)
                    route.fulfill(response=response, headers={**response.headers, **cors_headers(current)})
                    response.dispose()
            except Exception as error:
                audit.append({'event': 'guard_error', 'error': type(error).__name__})
                route.abort()
        for origin in origins(original):
            context.route(origin.rstrip('/') + '/**', redirect)
        audit.append({'event': 'context_guard_installed', 'mode': 'sync'})
        return context
    Browser.new_context = create
    try:
        yield
    finally:
        Browser.new_context = before
