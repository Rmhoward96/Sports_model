import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const SITE = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

// Minimal browser stand-ins: enough for top-level script evaluation, not rendering.
function fakeDom(page) {
  const el = () => ({ innerHTML: "", style: {}, dataset: {}, classList: { add() {}, remove() {}, toggle() {} },
    appendChild() {}, setAttribute() {}, addEventListener() {}, querySelector: () => null, querySelectorAll: () => [] });
  return {
    body: { dataset: { page }, appendChild() {}, classList: { add() {}, remove() {} } },
    head: { appendChild() {} }, documentElement: el(),
    createElement: el, createElementNS: el, getElementById: () => null,
    querySelector: () => null, querySelectorAll: () => [], addEventListener() {},
  };
}

// Copy objects from vm realm into host realm; non-cloneable values (functions, DOM stand-ins) returned raw.
function toHost(v) {
  if (typeof v !== "object" || v === null) {
    return v;  // primitives and strings
  }
  try {
    return structuredClone(v);
  } catch (e) {
    if (e?.name === "DataCloneError") {
      return v;
    }
    throw e;
  }
}

export function loadScripts(files, { page = "dashboard", storage = new Map() } = {}) {
  const localStorage = {
    getItem: (k) => (storage.has(k) ? storage.get(k) : null),
    setItem: (k, v) => storage.set(k, String(v)), removeItem: (k) => storage.delete(k),
  };
  const ctx = { console, URL, URLSearchParams, Intl, Date, Math, JSON, setTimeout, clearTimeout,
    setInterval: () => 0, fetch: async () => { throw new Error("no network in tests"); },
    localStorage, location: { search: "", href: "http://localhost/" }, history: { replaceState() {} } };
  ctx.window = ctx;
  ctx.__CA_TEST__ = true;            // js/boot.js skips boot() in tests
  ctx.document = fakeDom(page);
  vm.createContext(ctx);
  // Classic scripts share one global scope: concatenate so top-level const/let are visible to later files.
  const src = files.map((f) => fs.readFileSync(path.join(SITE, f), "utf8")).join("\n;\n");
  vm.runInContext(src + "\n;globalThis.__exports = { " + exportNames(src) + " };", ctx, { filename: files.join("+") });

  // Wrap plain function exports (not classes) to return values in host realm; keep non-function exports raw (live references).
  const exports = {};
  for (const [key, value] of Object.entries(ctx.__exports)) {
    if (typeof value === "function") {
      // Check if it's a class (not a plain function)
      const isClass = /^class\b/.test(Function.prototype.toString.call(value));
      if (isClass) {
        // Leave classes raw
        exports[key] = value;
      } else {
        // Wrap plain functions to return values in host realm
        exports[key] = (...args) => {
          const result = value(...args);
          // Handle both async and sync functions: if result is thenable, chain toHost.
          if (result && typeof result.then === "function") {
            return result.then(toHost);
          }
          return toHost(result);
        };
      }
    } else {
      // Non-function exports: keep raw (live references).
      exports[key] = value;
    }
  }

  // Keep localStorage as live reference (tests mutate/inspect it).
  return Object.assign(exports, { localStorage: ctx.localStorage });
}

// Every top-level function / const / let / class name, so tests can reach them.
function exportNames(src) {
  const names = new Set();
  for (const m of src.matchAll(/^(?:async\s+)?function\s+([A-Za-z_$][\w$]*)|^(?:const|let|class)\s+([A-Za-z_$][\w$]*)/gm)) {
    names.add(m[1] || m[2]);
  }
  return [...names].join(", ");
}
