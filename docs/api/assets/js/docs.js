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

  // Syntax highlighting (basic)
  function highlightCode() {
    document.querySelectorAll('code.language-python').forEach(function(block) {
      var html = block.innerHTML;

      // Order matters - do comments first, then strings, then others
      var patterns = [
        // Comments
        { regex: /(#.*)$/gm, class: 'token-comment' },
        // Triple-quoted strings
        { regex: /("""[\s\S]*?"""|'''[\s\S]*?''')/g, class: 'token-string' },
        // Double-quoted strings
        { regex: /("(?:[^"\\]|\\.)*")/g, class: 'token-string' },
        // Single-quoted strings
        { regex: /('(?:[^'\\]|\\.)*')/g, class: 'token-string' },
        // Keywords
        { regex: /\b(from|import|class|def|return|if|else|elif|for|while|try|except|finally|with|as|yield|lambda|and|or|not|in|is|True|False|None|async|await)\b/g, class: 'token-keyword' },
        // Built-ins
        { regex: /\b(print|len|range|str|int|float|list|dict|set|tuple|bool|type|isinstance|hasattr|getattr|setattr|open|super|self)\b/g, class: 'token-builtin' },
        // Numbers
        { regex: /\b(\d+\.?\d*)\b/g, class: 'token-number' },
        // Function definitions
        { regex: /\b(def\s+)(\w+)/g, replace: '<span class="token-keyword">$1</span><span class="token-function">$2</span>' },
        // Class definitions
        { regex: /\b(class\s+)(\w+)/g, replace: '<span class="token-keyword">$1</span><span class="token-class">$2</span>' },
      ];

      patterns.forEach(function(pattern) {
        if (pattern.replace) {
          html = html.replace(pattern.regex, pattern.replace);
        } else {
          html = html.replace(pattern.regex, '<span class="' + pattern.class + '">$1</span>');
        }
      });

      block.innerHTML = html;
    });
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

  // Search functionality (basic)
  function initSearch() {
    var searchInput = document.querySelector('.header-search input');
    if (!searchInput) return;

    searchInput.addEventListener('keydown', function(e) {
      if (e.key === 'Enter') {
        var query = this.value.trim().toLowerCase();
        if (query) {
          // For now, just scroll to first match on page
          var content = document.querySelector('.main');
          if (!content) return;

          // Simple text search in headings
          var headings = content.querySelectorAll('h1, h2, h3');
          for (var i = 0; i < headings.length; i++) {
            if (headings[i].textContent.toLowerCase().includes(query)) {
              headings[i].scrollIntoView({ behavior: 'smooth', block: 'start' });
              break;
            }
          }
        }
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
