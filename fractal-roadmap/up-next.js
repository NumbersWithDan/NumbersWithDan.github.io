/*
 * "Up next" banner for the Fractal Roadmap static export.
 * Reads progress from the app's localStorage entry and drives the app's own
 * UI (nav buttons, node buttons, milestone checkboxes) so React state stays
 * the single source of truth.
 *
 * ROADMAP mirrors the stage/node data compiled into the Next.js bundle.
 * If the roadmap content changes in a rebuild, regenerate it.
 */
(function () {
	"use strict";

	var STORAGE_KEY = "fractal-roadmap-progress-v1";
	var PRIORITY_RANK = { critical: 0, high: 1, medium: 2, optional: 3, advanced: 4 };

	var ROADMAP = {"stages":[{"number":0,"name":"Mathematical Reboot","nodeIds":["proof-writing","real-analysis","linear-algebra","early-fractals"]},{"number":1,"name":"Graduate Foundations","nodeIds":["topology","measure-theory","functional-analysis","dynamical-systems","complex-analysis","probability","abstract-algebra","fractal-bridge"]},{"number":2,"name":"Fractal Geometry Core","nodeIds":["ifs-core","falconer-core"]},{"number":3,"name":"Advanced Fractal Mathematics","nodeIds":["gmt-advanced","fractal-measures","fractal-analysis"]},{"number":4,"name":"Specialization","nodeIds":["specialization"]},{"number":5,"name":"Research Apprenticeship","nodeIds":["research-apprenticeship"]},{"number":6,"name":"Independent Research","nodeIds":["independent-research"]}],"nodes":[{"id":"proof-writing","name":"Proof Writing","stage":0,"priority":"high","prerequisites":[],"milestones":[["pw-quantifiers","Comfortable manipulating quantifiers"],["pw-sets","Comfortable proving set identities"],["pw-counter","Comfortable constructing counterexamples"],["pw-format","Can write clear theorem/proof format"],["pw-gaps","Can recognize gaps in informal arguments"]]},{"id":"real-analysis","name":"Real Analysis Refresh","stage":0,"priority":"critical","prerequisites":["proof-writing"],"milestones":[["ra-metric","Comfortable working with metric spaces"],["ra-compact","Can prove standard compactness results"],["ra-cauchy","Can work with Cauchy sequences and completeness"],["ra-uniform","Understand pointwise vs uniform convergence"],["ra-rudin","Can follow Rudin-style proofs comfortably"]]},{"id":"linear-algebra","name":"Linear Algebra Refresh","stage":0,"priority":"medium","prerequisites":[],"milestones":[["la-spaces","Comfortable with abstract vector spaces and maps"],["la-spectral","Recall spectral theorem and orthogonality"]]},{"id":"early-fractals","name":"Early Fractal Track","stage":0,"priority":"critical","prerequisites":[],"milestones":[["ef-examples","Can define and draw classical fractal examples"],["ef-ifs","Understand IFS and contraction mapping idea"],["ef-sim-dim","Compute similarity dimension for simple IFS"],["ef-box","Estimate box-counting dimension numerically"],["ef-hausdorff-intro","State introductory Hausdorff dimension definition"]]},{"id":"topology","name":"Topology","stage":1,"priority":"critical","prerequisites":["real-analysis"],"milestones":[["top-basics","Point-set topology basics"],["top-bases","Bases and subbases"],["top-product","Product topology"],["top-quotient","Quotient topology"],["top-compact","Compactness"],["top-connected","Connectedness"],["top-separation","Separation axioms"],["top-metric","Metric spaces in topological language"],["top-advanced","Selected advanced topics (Urysohn/Tychonoff)"],["top-exercises","Complete at least 25 substantial exercises"],["top-summary","Write a 5–10 page topology summary"],["top-ready","Topology Ready: read topology inside analysis papers"]]},{"id":"measure-theory","name":"Measure Theory","stage":1,"priority":"critical","prerequisites":["real-analysis"],"milestones":[["mt-sigma","Sigma algebras and measurable functions"],["mt-measures","Measures and outer measures"],["mt-lebesgue","Lebesgue integral"],["mt-convergence","Major convergence theorems"],["mt-product","Product measures"],["mt-fubini","Fubini / Tonelli"],["mt-signed","Signed measures"],["mt-rn","Radon–Nikodym"],["mt-radon","Radon measures"],["mt-weak","Weak convergence basics"],["mt-exercises","At least 30 meaningful exercises"],["mt-summary","Measure-theory summary document"],["mt-ready","Measure Theory Ready for Hausdorff measure"]]},{"id":"functional-analysis","name":"Functional Analysis","stage":1,"priority":"high","prerequisites":["real-analysis","linear-algebra"],"milestones":[["fa-banach","Work comfortably in Banach/Hilbert spaces"],["fa-theorems","Know the big three operator theorems"],["fa-dual","Use dual spaces and Hahn–Banach ideas"],["fa-compact","Understand compact operators at a working level"]]},{"id":"dynamical-systems","name":"Dynamical Systems","stage":1,"priority":"high","prerequisites":["real-analysis"],"milestones":[["dyn-logistic","Logistic map experiments"],["dyn-bifurcation","Bifurcation diagram"],["dyn-feigenbaum","Estimate Feigenbaum constant"],["dyn-lorenz","Lorenz attractor"],["dyn-henon","Hénon map"],["dyn-lyapunov","Lyapunov exponent estimation"],["dyn-symbolic","Symbolic dynamics experiment"],["dyn-theory","Explain chaos and sensitive dependence rigorously"]]},{"id":"complex-analysis","name":"Complex Analysis","stage":1,"priority":"medium","prerequisites":["real-analysis"],"milestones":[["ca-holomorphic","Fluent with holomorphic function theory"],["ca-conformal","Understand conformal maps and Möbius transformations"],["ca-montel","Normal families / Montel for dynamics readiness"]],"optional":true},{"id":"probability","name":"Probability","stage":1,"priority":"medium","prerequisites":["measure-theory"],"milestones":[["pr-mth","Measure-theoretic probability fluency"],["pr-cond","Conditional expectation comfort"],["pr-brownian","Introductory Brownian motion"]],"optional":true},{"id":"abstract-algebra","name":"Abstract Algebra","stage":1,"priority":"optional","prerequisites":[],"milestones":[["aa-basics","Refresh groups/rings/fields as needed"]],"optional":true},{"id":"fractal-bridge","name":"Fractal Bridge (Edgar)","stage":1,"priority":"critical","prerequisites":["early-fractals"],"milestones":[["fb-examples","Fractal examples chapter mastered"],["fb-metric","Metric topology for fractals"],["fb-topdim","Topological dimension ideas"],["fb-selfsim","Self-similarity chapter"],["fb-measure","Measure theory bridge chapters"],["fb-hausdorff","Hausdorff dimension in Edgar"]]},{"id":"ifs-core","name":"Iterated Function Systems","stage":2,"priority":"critical","prerequisites":["early-fractals","real-analysis"],"milestones":[["ifs-engine","Implement general IFS engine"],["ifs-cantor","Cantor set via IFS"],["ifs-sierpinski","Sierpiński triangle"],["ifs-fern","Barnsley fern"],["ifs-koch","Koch curve"],["ifs-custom","Custom IFS fractal"],["ifs-chaos","Chaos-game renderer"],["ifs-notes","Write IFS mathematical notes"]]},{"id":"falconer-core","name":"Core Fractal Geometry (Falconer)","stage":2,"priority":"critical","prerequisites":["measure-theory","topology","fractal-bridge"],"milestones":[["fg-hmeasure","Hausdorff measure"],["fg-hdim","Hausdorff dimension"],["fg-box","Box dimension"],["fg-packing","Packing dimension"],["fg-calc","Calculation techniques"],["fg-ifs","IFS theory"],["fg-osc","Open Set Condition"],["fg-mass","Mass distribution methods"],["fg-energy","Energy methods"],["fg-proj","Projection theory introduction"],["fg-affine","Self-affine sets introduction"],["fg-ready","Fractal Geometry Ready for papers"]]},{"id":"gmt-advanced","name":"Advanced Geometric Measure Theory","stage":3,"priority":"advanced","prerequisites":["falconer-core"],"milestones":[["gmt-densities","Densities and rectifiability overview"],["gmt-proj","Projection theorems at Mattila level"]],"advanced":true},{"id":"fractal-measures","name":"Fractal Measures","stage":3,"priority":"optional","prerequisites":["measure-theory","falconer-core"],"milestones":[["fm-bridge","Work through selected Edgar IPFM chapters"]],"optional":true,"advanced":true},{"id":"fractal-analysis","name":"Fractal Analysis","stage":3,"priority":"advanced","prerequisites":["functional-analysis","falconer-core"],"milestones":[["fa-dirichlet","Dirichlet / resistance forms overview"],["fa-laplacian","Laplacians on model fractals"],["fa-spectral","Spectral dimension ideas"]],"advanced":true},{"id":"specialization","name":"Specialization Branches","stage":4,"priority":"high","prerequisites":["falconer-core"],"milestones":[["sp-choose","Choose a primary specialization branch"],["sp-map","Map 5–10 key papers in that branch"],["sp-gap","Identify a concrete prerequisite gap list"]]},{"id":"research-apprenticeship","name":"Research Apprenticeship","stage":5,"priority":"critical","prerequisites":["falconer-core","specialization"],"milestones":[["rap-understood","First paper fully understood"],["rap-proof","First published proof reconstructed"],["rap-figure","First figure reproduced"],["rap-numeric","First numerical experiment replicated"],["rap-modify","First assumption modified"],["rap-novel","First potentially novel observation"]]},{"id":"independent-research","name":"Independent Research","stage":6,"priority":"critical","prerequisites":["research-apprenticeship"],"milestones":[["ir-question","Formulate a precise research question"],["ir-conjecture","State a nontrivial conjecture"],["ir-proof","Prove a nontrivial new (to you) result"],["ir-write","Write a research-quality manuscript draft"],["ir-feedback","Obtain external mathematical feedback"]]}]};

	var nodesById = {};
	ROADMAP.nodes.forEach(function (n) { nodesById[n.id] = n; });

	function readCompleted() {
		try {
			var raw = localStorage.getItem(STORAGE_KEY);
			return (raw && JSON.parse(raw).completedMilestones) || {};
		} catch (e) {
			return {};
		}
	}

	function nodeStatus(node, done) {
		var count = node.milestones.filter(function (m) { return done[m[0]]; }).length;
		if (count === 0) return "not-started";
		return count >= node.milestones.length ? "complete" : "in-progress";
	}

	function isRequired(node) {
		return !node.optional && !node.advanced;
	}

	// Same rule as the app: a prerequisite blocks only if it hasn't been started.
	function isBlocked(node, done) {
		return node.prerequisites.some(function (id) {
			var p = nodesById[id];
			return p && !p.optional && nodeStatus(p, done) === "not-started";
		});
	}

	function stageNodes(stage) {
		return stage.nodeIds.map(function (id) { return nodesById[id]; }).filter(Boolean);
	}

	function currentStage(done) {
		for (var i = 0; i < ROADMAP.stages.length; i++) {
			var required = stageNodes(ROADMAP.stages[i]).filter(isRequired);
			if (required.some(function (n) { return nodeStatus(n, done) !== "complete"; })) {
				return ROADMAP.stages[i];
			}
		}
		return null;
	}

	function computePlan() {
		var done = readCompleted();
		var stage = currentStage(done);
		if (!stage) return { stage: null, items: [] };

		var nodes = stageNodes(stage);
		var order = {};
		nodes.forEach(function (n, i) { order[n.id] = i; });

		var open = nodes.filter(function (n) {
			return isRequired(n) && nodeStatus(n, done) !== "complete";
		});
		var ready = open.filter(function (n) { return !isBlocked(n, done); });
		var candidates = (ready.length ? ready : open).sort(function (a, b) {
			var ap = nodeStatus(a, done) === "in-progress" ? 0 : 1;
			var bp = nodeStatus(b, done) === "in-progress" ? 0 : 1;
			return ap - bp ||
				PRIORITY_RANK[a.priority] - PRIORITY_RANK[b.priority] ||
				order[a.id] - order[b.id];
		});

		var items = candidates.map(function (n) {
			var next = n.milestones.find(function (m) { return !done[m[0]]; });
			var finished = n.milestones.filter(function (m) { return done[m[0]]; }).length;
			return {
				node: n,
				milestoneId: next[0],
				milestoneLabel: next[1],
				nodeDone: finished,
				nodeTotal: n.milestones.length
			};
		});

		var stageTotal = 0, stageDone = 0;
		nodes.filter(isRequired).forEach(function (n) {
			n.milestones.forEach(function (m) {
				stageTotal++;
				if (done[m[0]]) stageDone++;
			});
		});

		return { stage: stage, items: items, stageDone: stageDone, stageTotal: stageTotal };
	}

	// ---- Driving the app's UI -------------------------------------------------

	function waitFor(find, timeout) {
		return new Promise(function (resolve) {
			var start = Date.now();
			(function poll() {
				var el = find();
				if (el || Date.now() - start > (timeout || 3000)) return resolve(el || null);
				setTimeout(poll, 50);
			})();
		});
	}

	function findButton(scope, text) {
		return Array.prototype.find.call(document.querySelectorAll(scope + " button"), function (b) {
			return b.textContent.trim() === text;
		});
	}

	function findStageTrigger(number) {
		return Array.prototype.find.call(document.querySelectorAll('main [data-slot="accordion-trigger"]'), function (b) {
			return b.textContent.trim().indexOf("Stage " + number + " ") === 0;
		});
	}

	// Not every node appears in the skill tree; all of them are listed under their stage.
	function findNodeButton(node) {
		var direct = findButton("main", node.name);
		if (direct) return Promise.resolve(direct);
		var trigger = findStageTrigger(node.stage);
		if (!trigger) return Promise.resolve(null);
		if (trigger.getAttribute("aria-expanded") !== "true") trigger.click();
		return waitFor(function () {
			var id = trigger.getAttribute("aria-controls");
			var panel = id && document.getElementById(id);
			return panel && Array.prototype.find.call(panel.querySelectorAll("button"), function (b) {
				return b.textContent.trim() === node.name;
			});
		});
	}

	function goToMilestone(item) {
		var navBtn = findButton("nav", "Roadmap");
		if (navBtn) navBtn.click();
		return waitFor(function () { return findStageTrigger(item.node.stage); })
			.then(function () { return findNodeButton(item.node); })
			.then(function (nodeBtn) {
				if (nodeBtn) nodeBtn.click();
				return waitFor(function () { return document.getElementById(item.milestoneId); });
			})
			.then(function (input) {
				var row = input && input.closest("li");
				if (!row) return null;
				row.scrollIntoView({ behavior: "smooth", block: "center" });
				row.classList.remove("un-flash");
				void row.offsetWidth;
				row.classList.add("un-flash");
				return row;
			});
	}

	function markDone(item) {
		goToMilestone(item).then(function (row) {
			var box = row && row.querySelector('[role="checkbox"]');
			if (box && box.getAttribute("aria-checked") !== "true") box.click();
		});
	}

	// ---- Rendering ------------------------------------------------------------

	var STYLE = [
		"#up-next{max-width:80rem;margin:1.25rem auto 0;padding:0 1rem;}",
		"#up-next .un-card{border:1px solid var(--border);background:var(--card);color:var(--card-foreground,var(--foreground));border-radius:0.75rem;padding:1rem 1.25rem;display:flex;flex-wrap:wrap;gap:1rem 1.5rem;align-items:center;}",
		"#up-next .un-main{flex:1 1 22rem;min-width:0;}",
		"#up-next .un-eyebrow{font-size:0.7rem;letter-spacing:0.08em;text-transform:uppercase;color:var(--muted-foreground);}",
		"#up-next .un-title{font-size:1.15rem;font-weight:600;margin:0.2rem 0;}",
		"#up-next .un-sub{font-size:0.85rem;color:var(--muted-foreground);}",
		"#up-next .un-bar{height:4px;border-radius:2px;background:var(--muted);margin-top:0.6rem;overflow:hidden;max-width:24rem;}",
		"#up-next .un-bar > span{display:block;height:100%;background:var(--primary);}",
		"#up-next .un-actions{display:flex;gap:0.5rem;flex-wrap:wrap;}",
		"#up-next button{font:inherit;font-size:0.85rem;border-radius:0.5rem;padding:0.45rem 0.9rem;cursor:pointer;border:1px solid var(--border);background:transparent;color:inherit;}",
		"#up-next button.un-primary{background:var(--primary);color:var(--primary-foreground);border-color:var(--primary);}",
		"#up-next button:hover{opacity:0.85;}",
		"#up-next .un-also{flex-basis:100%;display:flex;flex-wrap:wrap;gap:0.4rem;align-items:center;font-size:0.8rem;color:var(--muted-foreground);}",
		"#up-next .un-also button{font-size:0.78rem;padding:0.25rem 0.6rem;}",
		"li.un-flash{animation:un-flash 2s ease-out;border-radius:0.5rem;}",
		"@keyframes un-flash{0%,40%{background:color-mix(in oklab,var(--primary) 25%,transparent);}100%{background:transparent;}}"
	].join("\n");

	function el(tag, attrs, children) {
		var node = document.createElement(tag);
		Object.keys(attrs || {}).forEach(function (k) {
			if (k === "onclick") node.addEventListener("click", attrs[k]);
			else if (k === "text") node.textContent = attrs[k];
			else node.setAttribute(k, attrs[k]);
		});
		(children || []).forEach(function (c) { if (c) node.appendChild(c); });
		return node;
	}

	var root = el("section", { id: "up-next", "aria-label": "Up next" });
	var lastRender = "";

	function render() {
		var plan = computePlan();
		var key = JSON.stringify([plan.stage && plan.stage.number, plan.items.map(function (i) { return i.milestoneId; }), plan.stageDone]);
		if (key === lastRender) return;
		lastRender = key;

		var card;
		if (!plan.stage) {
			card = el("div", { "class": "un-card" }, [
				el("div", { "class": "un-main" }, [
					el("div", { "class": "un-eyebrow", text: "Up next" }),
					el("div", { "class": "un-title", text: "Every required stage is complete." }),
					el("div", { "class": "un-sub", text: "Pick a specialization branch, a project, or a paper to reproduce." })
				])
			]);
		} else {
			var top = plan.items[0];
			var pct = plan.stageTotal ? Math.round((plan.stageDone / plan.stageTotal) * 100) : 0;
			var others = plan.items.slice(1, 4);

			card = el("div", { "class": "un-card" }, [
				el("div", { "class": "un-main" }, [
					el("div", { "class": "un-eyebrow", text: "Up next · Stage " + plan.stage.number + " — " + plan.stage.name }),
					el("div", { "class": "un-title", text: top.milestoneLabel }),
					el("div", { "class": "un-sub", text: top.node.name + " · milestone " + (top.nodeDone + 1) + " of " + top.nodeTotal + " · stage " + pct + "% complete (" + plan.stageDone + "/" + plan.stageTotal + ")" }),
					el("div", { "class": "un-bar" }, [el("span", { style: "width:" + pct + "%" })])
				]),
				el("div", { "class": "un-actions" }, [
					el("button", { "class": "un-primary", type: "button", text: "Go to it", onclick: function () { goToMilestone(top); } }),
					el("button", { type: "button", text: "Mark done", onclick: function () { markDone(top); } })
				]),
				others.length ? el("div", { "class": "un-also" }, [el("span", { text: "Also open in this stage:" })].concat(
					others.map(function (item) {
						return el("button", {
							type: "button",
							title: item.milestoneLabel,
							text: item.node.name + ": " + item.milestoneLabel,
							onclick: function () { goToMilestone(item); }
						});
					})
				)) : null
			]);
		}

		root.replaceChildren(card);
	}

	function mount() {
		var header = document.querySelector("header");
		if (!header || !header.parentNode) return;
		if (root.previousElementSibling !== header) header.after(root);
	}

	function init() {
		document.head.appendChild(el("style", { text: STYLE }));
		render();
		mount();

		// React may re-render the shell and drop foreign nodes; put the banner back.
		new MutationObserver(mount).observe(document.body, { childList: true, subtree: true });

		// The app persists every state change to localStorage; refresh when it does.
		var setItem = Storage.prototype.setItem;
		Storage.prototype.setItem = function (k) {
			setItem.apply(this, arguments);
			if (k === STORAGE_KEY) render();
		};
		window.addEventListener("storage", function (e) { if (e.key === STORAGE_KEY) render(); });
	}

	if (document.readyState === "complete") setTimeout(init, 0);
	else window.addEventListener("load", function () { setTimeout(init, 0); });
})();
