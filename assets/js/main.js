/*
	TXT by HTML5 UP
	html5up.net | @ajlkn
	Free for personal and commercial use under the CCA 3.0 license (html5up.net/license)
*/

(function($) {

	var	$window = $(window),
		$body = $('body'),
		$nav = $('#nav');

	// Breakpoints.
		breakpoints({
			xlarge:  [ '1281px',  '1680px' ],
			large:   [ '981px',   '1280px' ],
			medium:  [ '737px',   '980px'  ],
			small:   [ '361px',   '736px'  ],
			xsmall:  [ null,      '360px'  ]
		});

	// Play initial animations on page load.
		$window.on('load', function() {
			window.setTimeout(function() {
				$body.removeClass('is-preload');
			}, 100);
		});

	// Zen scroll fade-in for content sections.
		$window.on('load', function() {
			var fadeSelectors = '#main .box.highlight, #main .box.features, #main .box.feature, #main .box.blog, #main .box.post, #main .box.page-content, #main .container .row > .col-12 > section';
			var fadeElements = document.querySelectorAll(fadeSelectors);
			if (fadeElements.length === 0) {
				fadeElements = document.querySelectorAll('#main .box, #main > .container .row > *');
			}
			fadeElements.forEach(function(el, i) {
				el.classList.add('fade-in');
				el.style.transitionDelay = Math.min(i * 0.06, 0.3) + 's';
			});
			if ('IntersectionObserver' in window) {
				var observer = new IntersectionObserver(function(entries) {
					entries.forEach(function(entry) {
						if (entry.isIntersecting) {
							entry.target.classList.add('visible');
						}
					});
				}, { rootMargin: '0px 0px -40px 0px', threshold: 0.02 });
				fadeElements.forEach(function(el) { observer.observe(el); });
			} else {
				fadeElements.forEach(function(el) { el.classList.add('visible'); });
			}
		});

	// Dropdowns.
		$('#nav > ul').dropotron({
			mode: 'fade',
			noOpenerFade: true,
			speed: 300,
			alignment: 'center'
		});

	// Scrolly
		$('.scrolly').scrolly({
			speed: 1000,
			offset: function() { return $nav.height() - 5; }
		});

	// Nav.

		// Title Bar.
			$(
				'<div id="titleBar">' +
					'<a href="#navPanel" class="toggle"></a>' +
					'<span class="title">' + $('#logo').html() + '</span>' +
				'</div>'
			)
				.appendTo($body);

		// Panel.
			$(
				'<div id="navPanel">' +
					'<nav>' +
						$('#nav').navList() +
					'</nav>' +
				'</div>'
			)
				.appendTo($body)
				.panel({
					delay: 500,
					hideOnClick: true,
					hideOnSwipe: true,
					resetScroll: true,
					resetForms: true,
					side: 'left',
					target: $body,
					visibleClass: 'navPanel-visible'
				});

})(jQuery);