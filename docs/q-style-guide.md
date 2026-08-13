## Idiomatic Q Style

Write Q like a Q programmer, not a Python programmer using Q syntax.

- **Vectorize, don't loop.** If an operation works on atoms, it works on vectors. `2*x` doubles every element — no `each` needed. Reserve `each` for non-atomic functions.
- **Compose with adverbs, not temp variables.** Prefer `(+/)x` over `r:0; {r+:x}each x`. Prefer `{x*x} each x` over storing intermediates.
- **Keep functions short.** A Q function is typically 1-3 lines. If it's longer, break it into named helpers.
- **Use implicit args** (`x`, `y`, `z`) for short lambdas. Use explicit `{[a;b;c] ...}` for clarity with 2+ args or complex bodies.
- **Avoid global state.** Use functional style — input→output, no side effects.
- **Avoid `if[]` in expressions.** `if` is a statement with no return value. Use `$[cond;true;false]` for conditional expressions.
- **Avoid `do[]` and `while[]`.** Use Over (`/`) and Scan (`\`) instead. `n f/ x` replaces do-loops. `{cond}f/ x` replaces while-loops.
- **Flatten conditionals.** `$[c1;v1;c2;v2;c3;v3;default]` instead of nested `$[c1;v1;$[c2;v2;...]]`.
- **Embrace nulls.** Return `0N`, `0n`, or `(::)` for missing values — don't invent sentinel values.

---

## How Q Differs from Every Other Language

These are not edge cases — they affect nearly every function you write.

**Right-to-left evaluation, no precedence.** All operators bind right-to-left with equal precedence. There is no PEMDAS/BODMAS.
```
  WRONG: 2*3+4  → Q reads as 2*(3+4) = 14
  RIGHT: (2*3)+4 = 10
```
Parenthesize every sub-expression that assumes left-to-right or precedence ordering.

**`%` is division, not modulo.** This is the single most dangerous Python→Q mapping.
```
  Python: x % n    → Q: x mod n
  Python: x / y    → Q: x % y
  Python: x // y   → Q: x div y
  Python: x ** y   → Q: x xexp y   (returns float)
```

**No negative indexing.** `L -1` is subtraction (`L` minus `1`), not the last element.
```
  WRONG: x -1          → subtraction
  RIGHT: last x        → last element
  RIGHT: x count[x]-1  → last element via index
```

**Atoms and vectors are different types.** A single value is an atom; a list of one value is a vector. Many operations behave differently.
```
  "a"        → char atom, type -10h
  "abc"      → char vector (string), type 10h
  enlist "a" → one-element char vector, type 10h
  5          → long atom, type -7h
  enlist 5   → one-element long vector, type 7h
  count 5    → 1 (atom), but count "hello" → 5 (vector)
```
Use `(),x` or `enlist x` to promote an atom to a one-element list when needed.

**Out-of-bounds returns null, not an error.** `(1 2 3) 5` returns `0N`, silently. Check bounds explicitly if the algorithm depends on it.

**`=` is element-wise; `~` (Match) compares structures.**
```
  (1 2 3)=(1 2 4) → 1 1 0b   (boolean vector, NOT a single bool)
  (1 2 3)~(1 2 3) → 1b       (structural match, single bool)
  (1 2 3)~(1 2 4) → 0b
```
Never use `=` to check if two lists are the same. Use `~`.

---

## Python→Q Quick Reference

| Python | Q | Notes |
|--------|---|-------|
| `range(n)` | `til n` | |
| `len(x)` | `count x` | |
| `x[::-1]` | `reverse x` | |
| `sorted(x)` | `asc x` | `desc x` for descending |
| `set(x)` | `distinct x` | |
| `x.count(v)` | `sum x=v` | |
| `x.index(v)` | `x?v` | returns `count x` if not found |
| `sum(x)` | `sum x` | |
| `max(x)` | `max x` | |
| `abs(x)` | `abs x` | |
| `x % n == 0` | `0=x mod n` | `%` is division in Q! |
| `str(n)` | `string n` | returns char vector |
| `int(s)` | `"J"$s` | uppercase J for string parse |
| `float(s)` | `"F"$s` | uppercase F for string parse |
| `chr(n)` | `"c"$n` | |
| `ord(c)` | `"i"$c` | |
| `x[-1]` | `last x` | |
| `x[0]` | `first x` | |
| `"sep".join(L)` | `"sep" sv L` | |
| `s.split("sep")` | `"sep" vs s` | |
| `s.replace(a,b)` | `ssr[s;a;b]` | |
| `x if c else y` | `$[c;x;y]` | |
| `if/elif/else` | `$[c1;v1;c2;v2;dflt]` | flat chained conditional |
| `[f(i) for i in x]` | `f each x` | or just `f x` if f is atomic |
| `filter(p, x)` | `x where p x` | |
| `for` / `while` loop | `n f/ x` or `{cond}f/ x` | see Iteration below |
| `reduce(f, x)` | `f/ x` | Over: `(+/)1 2 3` = 6 |
| `accumulate(f, x)` | `f\ x` | Scan: `(+\)1 2 3` = 1 3 6 |
| `digits of n` | `10 vs n` | base conversion |
| `n from digits` | `10 sv d` | reverse base conversion |
| `zip(a, b)` | `flip(a;b)` | |
| `math.floor` | `floor x` | |
| `math.ceil` | `ceiling x` | |
| `None` | `(::)` | generic null; `0N` for null long, `0n` for null float |

---

## Iteration Without Loops

Q has no `for` or `while`. Use adverbs (iterators) instead:

- **Each** `f each x` or `f'x` — apply f to each element (like `map`)
- **Each Right/Left** `x f/: y` and `x f\: y` — cross-apply (nested loop)
- **Each Prior** `(-':)x` — pairwise differences (like `numpy.diff`); also `deltas x`
- **Over (reduce)** `f/ x` — fold: `(+/)1 2 3` gives `6`
- **Scan (accumulate)** `f\ x` — running fold: `(+\)1 2 3` gives `1 3 6`
- **Do N times** `n f/ x` — apply f repeatedly: `3 {x*2}/ 1` gives `8`
- **While** `{cond} f/ x` — repeat while true: `{x<100}{x*2}/ 1` gives `128`
- **Converge** `f/ x` (no left arg) — repeat until stable

Most built-in functions are already atomic (work on vectors): `2*1 2 3` gives `2 4 6`. Don't add `each` when the function already iterates.

---

## Syntax Traps

- **R-to-L eval in all expressions**: `x 0|y 0` parses as `x[0|y[0]]`. Parenthesize: `(x 0)|(y 0)`.
- **`x i j` is `x[i;j]`**, not `(x i) j`. Parenthesize for chaining: `(x i) j`.
- **Inline assignments can confuse the parser**: `(min s:sums d)>=0` — assign first: `s:sums d; (min s)>=0`.
- **Negative literal adjacency**: `1 -1` is the list `(1;-1)`, but it's ambiguous in expressions. Use `(1;-1)` explicitly.
- **No `return` keyword**: Use `:value` for early return: `{if[x<0; :0]; x*x}`.
- **Semicolons separate statements** inside braces. Last expression is the return value.
- **`$[...]` with even args and no match returns generic null** — not an error. Always include a default.
- **`and`/`or` are not short-circuit**. Use `$[cond1;cond2;0b]` for short-circuit logic.
- **`&` is min, `|` is max** — not logical AND/OR. `1&0` is `0` (min), `3|5` is `5` (max). For boolean vectors this happens to work, but `x>1 & all y` means `x > min(1; all y)`, not `(x>1) and (all y)`.
- **`/` is comment, not adverb, at start of token.** `*/x` is parsed as `*` then comment. Write `(*/)x` or `prd x`. Same for `+/x` → `(+/)x` or `sum x`.
- **No C-style string escapes.** `"\t"`, `"\n"` are invalid in Q. Use `"\t"` → `0x09`, `"\n"` → `"\n"` → `10h$10`. Or just use the char: `" "` for space.
- **`deltas` includes first element as-is.** `deltas 1 3 6` → `1 2 3` (first element is `1-0=1`). Use `1_deltas x` for pure pairwise differences.
- **`round` does not exist.** Use `"j"$x` to round to nearest long, or `0.01*"j"$100*x` for 2 decimal places.
- **`even` does not exist.** Use `0=x mod 2`.

---

## Type System

- **Default integer is long** (`-7h`). `til 5` produces longs.
- **Char atom vs string**: `"a"` is `-10h`, `"abc"` is `10h`. Use `(),x` to promote.
- **`"j"$"3"` returns 51** (ASCII of "3"), not 3. Use uppercase `"J"$enlist "3"` to parse a char as a number.
- **Booleans are `1b`/`0b`** and numeric: `sum 1 0 1 1b` = `3`.
- **Null values are typed**: `0N` (long), `0n` (float), `" "` (char), `` ` `` (symbol). Arithmetic propagates: `1+0N` = `0N`.
- **Empty lists lose type through `each`**: `f each ()` returns `()` (type 0h) even if f would return typed results.
- **`sum ()`** returns `()`, not `0`. Guard: `$[count x;sum x;0]`.
- **`raze ()`** returns `()`, not `""`. Guard with empty check.
- **`n#x` cycles**: `7#"ab"` = `"abababa"`. Use `n sublist x` for Python-style slicing.
- **String comparison**: `<`/`>` on different-length strings throws `'length`. Compare as symbols instead.

---

## Reserved Words and Variables

Never use Q builtins as local variable names — causes `'assign`:

`neg`, `type`, `string`, `max`, `min`, `sum`, `avg`, `count`, `first`, `last`,
`key`, `value`, `get`, `set`, `not`, `null`, `where`, `til`, `enlist`, `raze`,
`flip`, `asc`, `desc`, `distinct`, `group`, `in`, `like`, `within`, `differ`,
`except`, `inter`, `union`, `read0`, `read1`, `ss`, `sv`, `vs`, `ssr`, `abs`,
`floor`, `ceiling`, `deltas`, `sums`, `prds`, `prd`

Use descriptive names instead: `negvals`, `posvals`, `cnt`, `res`, `vals`.

---

## IPC / Test Harness

- **Python ints arrive as Q longs** (`-7h`), not ints (`-6h`). Filter with `(-7h)=type each x`.
- **Python floats arrive as Q floats** (`-9h`), not reals (`-8h`).
- **Function arity must match tests**: If tests call `candidate(a, b)`, define `{[x;y] ...}`. If `candidate(lst)`, define `{[x] ...}`. Mismatch → `'rank`.
- **Return `(::)` for Python `None`**. Tests comparing `== None` expect generic null.
- **Inner lambdas** don't close over sibling locals. Capture via projection: `{[x] {[f;y] f y}[mylocal;] each x}`.
- **`,:` vs `,`**: `x,:(pair)` appends one item; `x:x,(pair)` concatenates. Wrong one scrambles list structure.
- **Augmented assign trap**: `aand:b` creates variable `aand`, doesn't mean `a:a and b`. Spell it out.
