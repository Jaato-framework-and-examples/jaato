/**
 * jaato API Documentation - Interactive functionality
 */

(function() {
  'use strict';

  // Copy to clipboard functionality
  function initCopyButtons() {
    document.querySelectorAll('.code-block').forEach(function(block) {
      // Skip if already has a copy button
      if (block.querySelector('.copy-btn')) return;

      var btn = document.createElement('button');
      btn.className = 'copy-btn';
      btn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg><span>Copy</span>';

      btn.addEventListener('click', function() {
        var code = block.querySelector('code');
        var text = code ? code.textContent : '';

        navigator.clipboard.writeText(text).then(function() {
          btn.classList.add('copied');
          btn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg><span>Copied!</span>';

          setTimeout(function() {
            btn.classList.remove('copied');
            btn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg><span>Copy</span>';
          }, 2000);
        }).catch(function(err) {
          console.error('Failed to copy:', err);
        });
      });

      block.appendChild(btn);
    });
  }

  // Syntax highlighting disabled - the naive regex approach corrupts HTML
  // by matching keywords inside already-replaced class attributes.
  // Consider using a proper library like Prism.js or highlight.js instead.
  function highlightCode() {
    // Disabled
  }

  // Active sidebar navigation
  function initSidebarNav() {
    var currentPath = window.location.pathname;

    document.querySelectorAll('.sidebar-nav a').forEach(function(link) {
      var href = link.getAttribute('href');
      if (href && currentPath.endsWith(href.replace(/^\.\.?\//, ''))) {
        link.classList.add('active');
      }
    });
  }

  // Smooth scroll for anchor links
  function initAnchorLinks() {
    document.querySelectorAll('a[href^="#"]').forEach(function(link) {
      link.addEventListener('click', function(e) {
        var targetId = this.getAttribute('href').slice(1);
        var target = document.getElementById(targetId);

        if (target) {
          e.preventDefault();
          target.scrollIntoView({ behavior: 'smooth', block: 'start' });
          history.pushState(null, null, '#' + targetId);
        }
      });
    });
  }

  // Add anchor links to headings
  function initHeadingAnchors() {
    document.querySelectorAll('.panel-explanation h2[id], .panel-explanation h3[id]').forEach(function(heading) {
      var id = heading.getAttribute('id');
      var anchor = document.createElement('a');
      anchor.className = 'anchor-link';
      anchor.href = '#' + id;
      anchor.innerHTML = '#';
      heading.appendChild(anchor);
    });
  }

  // Generate "On This Page" navigation and track scroll
  function initOnThisPage() {
    var sidebar = document.querySelector('.sidebar');
    if (!sidebar) return;

    // Guard against double initialization
    if (sidebar.getAttribute('data-on-this-page-init')) return;
    sidebar.setAttribute('data-on-this-page-init', 'true');

    // Find all h2 headings with IDs in the main content
    var headings = document.querySelectorAll('.main h2[id]');
    if (headings.length < 2) return; // Don't show for pages with few sections

    // Create the "On This Page" section
    var section = document.createElement('div');
    section.className = 'sidebar-section on-this-page';
    section.innerHTML = '<div class="sidebar-title">On This Page</div>';

    var nav = document.createElement('ul');
    nav.className = 'sidebar-nav on-this-page-nav';

    headings.forEach(function(heading) {
      var li = document.createElement('li');
      var link = document.createElement('a');
      link.href = '#' + heading.id;
      link.textContent = heading.textContent.replace(/#$/, '').trim(); // Remove anchor #
      link.setAttribute('data-target', heading.id);
      li.appendChild(link);
      nav.appendChild(li);
    });

    section.appendChild(nav);

    // Insert at the top of sidebar
    sidebar.insertBefore(section, sidebar.firstChild);

    // Scroll spy: track which section is currently visible
    var links = nav.querySelectorAll('a');
    var headerHeight = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--header-height')) || 60;

    function updateActiveLink() {
      var currentId = null;

      // Use getBoundingClientRect for accurate viewport-relative positions
      // Find the last heading that has scrolled past the top of the viewport
      for (var i = 0; i < headings.length; i++) {
        var rect = headings[i].getBoundingClientRect();
        // Consider heading "active" when it's at or above the header + some offset
        if (rect.top <= headerHeight + 50) {
          currentId = headings[i].id;
        }
      }

      // If nothing matched yet, use the first heading
      if (!currentId && headings.length > 0) {
        currentId = headings[0].id;
      }

      // Update active states
      links.forEach(function(link) {
        if (link.getAttribute('data-target') === currentId) {
          link.classList.add('active');
        } else {
          link.classList.remove('active');
        }
      });
    }

    // Throttle scroll events for performance
    var ticking = false;
    window.addEventListener('scroll', function() {
      if (!ticking) {
        window.requestAnimationFrame(function() {
          updateActiveLink();
          ticking = false;
        });
        ticking = true;
      }
    });

    // Initial update
    updateActiveLink();
  }

  // Search functionality with Lunr.js
  var searchState = {
    index: null,
    documents: null,
    loaded: false,
    resultsContainer: null
  };

  function initSearch() {
    var searchInput = document.querySelector('.header-search input');
    if (!searchInput) return;

    // Create results container
    var searchContainer = document.querySelector('.header-search');
    searchState.resultsContainer = document.createElement('div');
    searchState.resultsContainer.className = 'search-results';
    searchContainer.appendChild(searchState.resultsContainer);

    // Load search index
    loadSearchIndex();

    // Handle input changes (debounced)
    var searchTimeout;
    searchInput.addEventListener('input', function() {
      clearTimeout(searchTimeout);
      var query = this.value.trim();

      if (query.length < 2) {
        hideSearchResults();
        return;
      }

      searchTimeout = setTimeout(function() {
        performSearch(query);
      }, 300);
    });

    // Handle focus
    searchInput.addEventListener('focus', function() {
      if (this.value.trim().length >= 2) {
        performSearch(this.value.trim());
      }
    });

    // Handle keyboard navigation
    searchInput.addEventListener('keydown', function(e) {
      if (e.key === 'Escape') {
        hideSearchResults();
        this.blur();
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        focusFirstResult();
      }
    });

    // Hide results when clicking outside
    document.addEventListener('click', function(e) {
      if (!searchContainer.contains(e.target)) {
        hideSearchResults();
      }
    });

    // Keyboard shortcut: / to focus search
    document.addEventListener('keydown', function(e) {
      if (e.key === '/' && document.activeElement !== searchInput) {
        e.preventDefault();
        searchInput.focus();
      }
    });
  }

  function loadSearchIndex() {
    if (searchState.loaded) return;

    // Check if Lunr is available
    if (typeof lunr === 'undefined') {
      console.warn('Lunr.js not loaded. Search functionality disabled.');
      return;
    }

    // Determine base path for loading index
    var basePath = getBasePath();

    fetch(basePath + 'assets/js/search-index.json')
      .then(function(response) {
        if (!response.ok) throw new Error('Failed to load search index');
        return response.json();
      })
      .then(function(documents) {
        searchState.documents = documents;

        // Build Lunr index
        searchState.index = lunr(function() {
          this.ref('id');
          this.field('title', { boost: 10 });
          this.field('headings', { boost: 5 });
          this.field('content');

          documents.forEach(function(doc) {
            this.add(doc);
          }, this);
        });

        searchState.loaded = true;
        console.log('Search index loaded:', documents.length, 'documents');
      })
      .catch(function(error) {
        console.error('Error loading search index:', error);
      });
  }

  function getBasePath() {
    // Determine base path based on current location
    var path = window.location.pathname;

    console.log('[getBasePath] pathname:', path);

    // Find the /api/ directory in the path - this is our base
    var apiIndex = path.indexOf('/api/');
    if (apiIndex === -1) {
      // Fallback: not in expected structure
      console.warn('[getBasePath] /api/ not found in path, using root');
      return './';
    }

    // Get the path after /api/
    var pathAfterApi = path.substring(apiIndex + 5); // +5 for '/api/'

    console.log('[getBasePath] pathAfterApi:', pathAfterApi);

    // Remove filename to get directory path after /api/
    var lastSlash = pathAfterApi.lastIndexOf('/');
    var dirAfterApi = lastSlash > 0 ? pathAfterApi.substring(0, lastSlash) : '';

    console.log('[getBasePath] dirAfterApi:', dirAfterApi);

    // Count depth (number of directory levels after /api/)
    var depth = dirAfterApi === '' ? 0 : (dirAfterApi.match(/\//g) || []).length + 1;

    console.log('[getBasePath] depth:', depth);

    // If at root of /api/, no traversal needed
    if (depth === 0) {
      console.log('[getBasePath] returning: ./');
      return './';
    }

    // Build relative path to go up to /api/ root
    var base = '';
    for (var i = 0; i < depth; i++) {
      base += '../';
    }
    console.log('[getBasePath] returning:', base);
    return base;
  }

  function performSearch(query) {
    if (!searchState.loaded || !searchState.index) {
      showSearchResults([{ title: 'Loading index...', snippet: 'Please wait...' }], query, true);
      return;
    }

    try {
      // Perform search with wildcards for partial matching
      var searchQuery = query.split(/\s+/).map(function(term) {
        return term + '*';
      }).join(' ');

      var results = searchState.index.search(searchQuery);

      // Limit to top 10 results
      results = results.slice(0, 10);

      // Enrich results with document data
      var enrichedResults = results.map(function(result) {
        var doc = searchState.documents.find(function(d) {
          return d.id === result.ref;
        });
        return {
          title: doc.title,
          url: doc.url,
          section: doc.section,
          snippet: doc.snippet,
          score: result.score
        };
      });

      showSearchResults(enrichedResults, query);
    } catch (error) {
      console.error('Search error:', error);
      showSearchResults([{ title: 'Search error', snippet: 'Please try a different query' }], query, true);
    }
  }

  function showSearchResults(results, query, isMessage) {
    var container = searchState.resultsContainer;
    container.innerHTML = '';
    container.style.display = 'block';

    if (results.length === 0) {
      container.innerHTML = '<div class="search-result search-no-results">No results found for "' + escapeHtml(query) + '"</div>';
      return;
    }

    if (isMessage) {
      container.innerHTML = '<div class="search-result">' + results[0].title + '</div>';
      return;
    }

    var basePath = getBasePath();

    results.forEach(function(result, index) {
      var resultEl = document.createElement('a');
      resultEl.className = 'search-result';
      // Add search query to URL for highlighting on target page
      resultEl.href = basePath + result.url + '?highlight=' + encodeURIComponent(query);
      resultEl.setAttribute('data-index', index);

      var titleEl = document.createElement('div');
      titleEl.className = 'search-result-title';
      titleEl.textContent = result.title;

      var sectionEl = document.createElement('div');
      sectionEl.className = 'search-result-section';
      sectionEl.textContent = result.section;

      var snippetEl = document.createElement('div');
      snippetEl.className = 'search-result-snippet';
      snippetEl.textContent = result.snippet;

      resultEl.appendChild(titleEl);
      resultEl.appendChild(sectionEl);
      resultEl.appendChild(snippetEl);

      // Handle keyboard navigation
      resultEl.addEventListener('keydown', function(e) {
        if (e.key === 'ArrowDown') {
          e.preventDefault();
          focusNextResult(index);
        } else if (e.key === 'ArrowUp') {
          e.preventDefault();
          focusPrevResult(index);
        } else if (e.key === 'Escape') {
          e.preventDefault();
          hideSearchResults();
          document.querySelector('.header-search input').focus();
        }
      });

      container.appendChild(resultEl);
    });
  }

  function hideSearchResults() {
    if (searchState.resultsContainer) {
      searchState.resultsContainer.style.display = 'none';
    }
  }

  function focusFirstResult() {
    var firstResult = searchState.resultsContainer.querySelector('.search-result');
    if (firstResult && firstResult.focus) {
      firstResult.focus();
    }
  }

  function focusNextResult(currentIndex) {
    var results = searchState.resultsContainer.querySelectorAll('.search-result');
    if (currentIndex < results.length - 1) {
      results[currentIndex + 1].focus();
    }
  }

  function focusPrevResult(currentIndex) {
    if (currentIndex > 0) {
      var results = searchState.resultsContainer.querySelectorAll('.search-result');
      results[currentIndex - 1].focus();
    } else {
      document.querySelector('.header-search input').focus();
    }
  }

  function escapeHtml(text) {
    var div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  // Highlight search terms on page load
  var highlightState = {
    currentIndex: 0,
    totalMatches: 0,
    matches: []
  };

  function initSearchHighlight() {
    // Check if there's a highlight parameter in URL
    var urlParams = new URLSearchParams(window.location.search);
    var highlightQuery = urlParams.get('highlight');

    if (!highlightQuery) return;

    console.log('[Search Highlight] Query:', highlightQuery);

    // Get main content area
    var mainContent = document.querySelector('.main');
    if (!mainContent) {
      console.warn('[Search Highlight] No .main element found');
      return;
    }

    // Split query into terms
    var terms = highlightQuery.toLowerCase().split(/\s+/).filter(function(term) {
      return term.length > 1; // Ignore single characters
    });

    console.log('[Search Highlight] Search terms:', terms);

    if (terms.length === 0) {
      console.warn('[Search Highlight] No valid terms after filtering');
      return;
    }

    // Find and highlight all matching text nodes
    highlightTermsInElement(mainContent, terms);

    // Get all highlights
    highlightState.matches = document.querySelectorAll('.search-highlight');
    highlightState.totalMatches = highlightState.matches.length;

    console.log('[Search Highlight] Total matches found:', highlightState.totalMatches);

    // Always show navigation bar (even if no matches)
    showHighlightNavigation(highlightQuery);

    if (highlightState.totalMatches > 0) {
      // Mark first as current
      highlightState.currentIndex = 0;
      highlightState.matches[0].classList.add('current');

      // Scroll to first highlight after a brief delay
      setTimeout(function() {
        scrollToCurrentHighlight();
      }, 100);
    } else {
      console.warn('[Search Highlight] No matches found for terms:', terms);
    }
  }

  function showHighlightNavigation(query) {
    // Create navigation bar
    var nav = document.createElement('div');
    nav.className = 'highlight-nav';

    var counterHtml = '';
    var buttonsHtml = '';

    if (highlightState.totalMatches > 0) {
      counterHtml = '<span class="highlight-nav-counter">' +
        '<span class="current-index">1</span> of <span class="total-matches">' + highlightState.totalMatches + '</span>' +
        '</span>';
      buttonsHtml =
        '<button class="highlight-nav-btn" id="highlight-prev" title="Previous match (P or ↑)">↑</button>' +
        '<button class="highlight-nav-btn" id="highlight-next" title="Next match (N or ↓)">↓</button>';
    } else {
      counterHtml = '<span class="highlight-nav-counter highlight-no-matches">No matches found on this page</span>';
    }

    nav.innerHTML =
      '<div class="highlight-nav-content">' +
        '<span class="highlight-nav-query">Highlighting: <strong>' + escapeHtml(query) + '</strong></span>' +
        counterHtml +
        '<div class="highlight-nav-buttons">' +
          buttonsHtml +
          '<button class="highlight-nav-btn" id="highlight-close" title="Close (ESC)">✕</button>' +
        '</div>' +
      '</div>';

    document.body.appendChild(nav);

    // Add event listeners
    if (highlightState.totalMatches > 0) {
      document.getElementById('highlight-prev').addEventListener('click', navigateToPrevHighlight);
      document.getElementById('highlight-next').addEventListener('click', navigateToNextHighlight);
    }
    document.getElementById('highlight-close').addEventListener('click', closeHighlightNavigation);

    // Keyboard shortcuts (only if we have matches)
    if (highlightState.totalMatches > 0) {
      document.addEventListener('keydown', handleHighlightNavigation);
    } else {
      // Only listen for ESC to close
      document.addEventListener('keydown', function handleEsc(e) {
        if (e.key === 'Escape') {
          closeHighlightNavigation();
          document.removeEventListener('keydown', handleEsc);
        }
      });
    }
  }

  function handleHighlightNavigation(e) {
    // Don't interfere if user is typing in an input
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    if (e.key === 'n' || e.key === 'N' || e.key === 'ArrowDown') {
      e.preventDefault();
      navigateToNextHighlight();
    } else if (e.key === 'p' || e.key === 'P' || e.key === 'ArrowUp') {
      e.preventDefault();
      navigateToPrevHighlight();
    } else if (e.key === 'Escape') {
      closeHighlightNavigation();
    }
  }

  function navigateToNextHighlight() {
    if (highlightState.totalMatches === 0) return;

    // Remove current class from current match
    highlightState.matches[highlightState.currentIndex].classList.remove('current');

    // Move to next (wrap around)
    highlightState.currentIndex = (highlightState.currentIndex + 1) % highlightState.totalMatches;

    // Add current class to new match
    highlightState.matches[highlightState.currentIndex].classList.add('current');

    // Update counter
    updateHighlightCounter();

    // Scroll to match
    scrollToCurrentHighlight();
  }

  function navigateToPrevHighlight() {
    if (highlightState.totalMatches === 0) return;

    // Remove current class from current match
    highlightState.matches[highlightState.currentIndex].classList.remove('current');

    // Move to previous (wrap around)
    highlightState.currentIndex = (highlightState.currentIndex - 1 + highlightState.totalMatches) % highlightState.totalMatches;

    // Add current class to new match
    highlightState.matches[highlightState.currentIndex].classList.add('current');

    // Update counter
    updateHighlightCounter();

    // Scroll to match
    scrollToCurrentHighlight();
  }

  function updateHighlightCounter() {
    var counter = document.querySelector('.current-index');
    if (counter) {
      counter.textContent = highlightState.currentIndex + 1;
    }
  }

  function scrollToCurrentHighlight() {
    var current = highlightState.matches[highlightState.currentIndex];
    if (current) {
      current.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }

  function closeHighlightNavigation() {
    // Remove navigation bar
    var nav = document.querySelector('.highlight-nav');
    if (nav) {
      nav.remove();
    }

    // Remove all highlights
    highlightState.matches.forEach(function(match) {
      var text = document.createTextNode(match.textContent);
      match.parentNode.replaceChild(text, match);
    });

    // Remove keyboard listener
    document.removeEventListener('keydown', handleHighlightNavigation);

    // Clear state
    highlightState.matches = [];
    highlightState.totalMatches = 0;
    highlightState.currentIndex = 0;

    // Clean up URL (remove highlight parameter)
    var url = new URL(window.location);
    url.searchParams.delete('highlight');
    window.history.replaceState({}, '', url);
  }

  function highlightTermsInElement(element, terms) {
    // Don't highlight in script tags, style tags, or code blocks
    var skipTags = ['SCRIPT', 'STYLE', 'CODE', 'PRE'];
    var matchCount = 0;

    function walkTextNodes(node) {
      if (node.nodeType === 3) { // Text node
        var parent = node.parentNode;
        if (parent && skipTags.indexOf(parent.tagName) !== -1) {
          return;
        }

        var text = node.textContent;
        var lowerText = text.toLowerCase();
        var hasMatch = false;

        // Check if any term matches
        for (var i = 0; i < terms.length; i++) {
          if (lowerText.indexOf(terms[i]) !== -1) {
            hasMatch = true;
            break;
          }
        }

        if (hasMatch) {
          // Create highlighted version
          var fragment = document.createDocumentFragment();
          var lastIndex = 0;

          // Build regex pattern from all terms
          var pattern = terms.map(function(term) {
            return term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); // Escape regex chars
          }).join('|');

          var regex = new RegExp('(' + pattern + ')', 'gi');
          var match;

          while ((match = regex.exec(text)) !== null) {
            // Add text before match
            if (match.index > lastIndex) {
              fragment.appendChild(document.createTextNode(text.substring(lastIndex, match.index)));
            }

            // Add highlighted match
            var mark = document.createElement('mark');
            mark.className = 'search-highlight';
            mark.textContent = match[0];
            fragment.appendChild(mark);
            matchCount++;

            lastIndex = match.index + match[0].length;
          }

          // Add remaining text
          if (lastIndex < text.length) {
            fragment.appendChild(document.createTextNode(text.substring(lastIndex)));
          }

          // Replace text node with highlighted version
          if (fragment.childNodes.length > 0) {
            parent.replaceChild(fragment, node);
          }
        }
      } else if (node.nodeType === 1 && node.childNodes) { // Element node
        // Don't process script, style, code blocks
        if (skipTags.indexOf(node.tagName) === -1) {
          // Process child nodes (iterate backwards to avoid issues with DOM changes)
          var children = Array.prototype.slice.call(node.childNodes);
          for (var i = 0; i < children.length; i++) {
            walkTextNodes(children[i]);
          }
        }
      }
    }

    walkTextNodes(element);
    console.log('[Search Highlight] Created', matchCount, 'highlight elements');
  }

  // Mobile menu toggle
  function initMobileMenu() {
    var menuBtn = document.querySelector('.mobile-menu-btn');
    var sidebar = document.querySelector('.sidebar');

    if (menuBtn && sidebar) {
      menuBtn.addEventListener('click', function() {
        sidebar.classList.toggle('open');
      });
    }
  }

  // Initialize on DOM ready
  function init() {
    initCopyButtons();
    highlightCode();
    initSidebarNav();
    initAnchorLinks();
    initHeadingAnchors();
    initOnThisPage();
    initSearch();
    initSearchHighlight();
    initMobileMenu();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
