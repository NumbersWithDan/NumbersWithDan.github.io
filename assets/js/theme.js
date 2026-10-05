/*
 * Light/dark theme + compact site header.
 * Loaded in <head> so the theme class is set before first paint.
 * Uses the same "theme" localStorage key as the Fractal Roadmap tracker,
 * so the choice carries across both.
 */
(function () {
	"use strict";

	var KEY = "theme";
	var root = document.documentElement;
	var media = window.matchMedia("(prefers-color-scheme: dark)");

	function stored() {
		try { return localStorage.getItem(KEY); } catch (e) { return null; }
	}

	function resolve() {
		var s = stored();
		if (s === "light" || s === "dark") return s;
		return media.matches ? "dark" : "light";
	}

	function apply(theme) {
		root.classList.remove("light", "dark");
		root.classList.add(theme);
		root.style.colorScheme = theme;
	}

	function toggle() {
		var next = root.classList.contains("dark") ? "light" : "dark";
		try { localStorage.setItem(KEY, next); } catch (e) {}
		apply(next);
	}

	apply(resolve());

	media.addEventListener("change", function () { apply(resolve()); });
	window.addEventListener("storage", function (e) { if (e.key === KEY) apply(resolve()); });

	var SUN = '<svg class="icon-sun" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/></svg>';
	var MOON = '<svg class="icon-moon" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/></svg>';

	function themeButton() {
		var btn = document.createElement("button");
		btn.type = "button";
		btn.className = "theme-toggle";
		btn.setAttribute("aria-label", "Toggle light/dark theme");
		btn.title = "Toggle light/dark theme";
		btn.innerHTML = SUN + MOON;
		btn.addEventListener("click", toggle);
		return btn;
	}

	document.addEventListener("DOMContentLoaded", function () {
		var logo = document.getElementById("logo");
		var name = logo ? logo.textContent.replace(/[\s\-–—]+$/, "").trim() : "Numbers With Dan";
		var href = logo ? logo.getAttribute("href") : "/";

		var nav = document.getElementById("nav");
		if (nav) {
			var brand = document.createElement("a");
			brand.className = "site-brand";
			brand.href = href;
			brand.innerHTML = '<span class="site-brand-name"></span><span class="site-brand-tag">Data · Math · Sports</span>';
			brand.firstChild.textContent = name;
			nav.insertBefore(brand, nav.firstChild);
			nav.appendChild(themeButton());
		}

		// #titleBar is built by main.js at the end of <body>, before this fires.
		var bar = document.getElementById("titleBar");
		if (bar) {
			var title = bar.querySelector(".title");
			if (title) title.textContent = name;
			bar.appendChild(themeButton());
		}
	});
})();
