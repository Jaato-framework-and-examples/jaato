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

    // Remove filename from path to get directory
    var dir = path.substring(0, path.lastIndexOf('/'));

    // Count directory depth (number of slashes after removing leading slash)
    var cleanDir = dir.replace(/^\//, ''); // Remove leading slash
    var depth = cleanDir === '' ? 0 : (cleanDir.match(/\//g) || []).length + 1;

    // If at root (index.html), no traversal needed
    if (depth === 0) {
      return './';
    }

    // Build relative path to go up to root
    var base = '';
    for (var i = 0; i < depth; i++) {
      base += '../';
    }
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
      resultEl.href = basePath + result.url;
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
    initMobileMenu();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
