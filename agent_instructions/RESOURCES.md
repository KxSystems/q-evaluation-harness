# Q/kdb+ Reference Resources for Python→Q Translation Tool

## Primary References (feed to Claude Code as context)

### Q for Mortals (Version 3) — Full book, free online
- **URL**: https://code.kx.com/q4m3/
- **Chapters**:
  - Ch 0: Overview — https://code.kx.com/q4m3/0_Overview/
  - Ch 1: Q Shock and Awe — https://code.kx.com/q4m3/1_Q_Shock_and_Awe/
  - Ch 2: Basic Data Types — https://code.kx.com/q4m3/2_Basic_Data_Types_Atoms/
  - Ch 3: Lists — https://code.kx.com/q4m3/3_Lists/
  - Ch 4: Operators — https://code.kx.com/q4m3/4_Operators/
  - Ch 5: Dictionaries — https://code.kx.com/q4m3/5_Dictionaries/
  - Ch 6: Functions — https://code.kx.com/q4m3/6_Functions/
  - Ch 7: Transforming Data — https://code.kx.com/q4m3/7_Transforming_Data/
  - Ch 8: Tables — https://code.kx.com/q4m3/8_Tables/
  - Ch 9: Queries (q-sql) — https://code.kx.com/q4m3/9_Queries_q-sql/
  - Ch 10: Execution Control — https://code.kx.com/q4m3/10_Execution_Control/
  - Ch 11: I/O — https://code.kx.com/q4m3/11_IO/
  - Ch 14: kdb+ Database — https://code.kx.com/q4m3/14_Introduction_to_Kdb+/
  - Appendix A: Built-in Functions — https://code.kx.com/q4m3/A_Built-in_Functions/
- **Why**: Idiomatic Q patterns, the way Q programmers actually write code. Essential for producing output that matches Q-HumanEval test expectations.

### Q Language Reference Card
- **URL**: https://code.kx.com/q/ref/
- **Why**: Complete listing of all keywords, operators, iterators, datatypes, and namespaces. The definitive quick reference for Q semantics.

### Q Syntax Reference
- **URL**: https://code.kx.com/q/basics/syntax/
- **Why**: Formal syntax rules — precedence, evaluation order, notation for lists/tables/functions, prefix/infix/postfix, iterators.

### Q Datatypes
- **URL**: https://code.kx.com/q/basics/datatypes/
- **Why**: Complete type system — critical for correct type casting in translation.

## Python↔Q Translation Examples (directly relevant)

### Examples from Python (on code.kx.com)
- **Basic**: https://code.kx.com/q/learn/python/examples/
- **Arrays**: https://code.kx.com/q/learn/python/examples/array/
- **Lists**: https://code.kx.com/q/learn/python/examples/list/
- **Strings**: https://code.kx.com/q/learn/python/examples/string/
- **Dictionaries**: https://code.kx.com/q/learn/python/examples/dict/
- **Why**: Side-by-side Python and Q solutions. This IS the translation task. Claude Code should study these closely.

## Secondary References

### Q by Topic
- **URL**: https://code.kx.com/q/basics/by-topic/
- **Why**: Organized reference by concept area.

### Q Function Notation
- **URL**: https://code.kx.com/q/basics/function-notation/
- **Why**: Lambda syntax, implicit args (x;y;z), closures.

### qSQL Queries
- **URL**: https://code.kx.com/q/basics/qsql/
- **Why**: select/exec/update/delete syntax for table operations.

### Iteration (adverbs/iterators)
- **URL**: https://code.kx.com/q/basics/iteration/
- **Why**: each, over, scan, each-prior, each-left, each-right — the key to translating Python loops.

### Joins
- **URL**: https://code.kx.com/q/basics/joins/
- **Why**: aj, lj, ij, uj, wj — Q's join types for table operations.

### Programming Idioms
- **URL**: https://code.kx.com/q/kb/programming-idioms/
- **Why**: Common Q patterns and idioms.

### Q Phrasebook
- **URL**: https://code.kx.com/phrases/
- **Why**: Collection of Q phrases and patterns for common operations.

### QIdioms (FinnAPL Idiom Library for Q)
- **URL**: https://code.kx.com/phrases/wikipage/
- **Why**: Hundreds of Q patterns ported from the FinnAPL idiom library. Dense but very useful for understanding idiomatic Q patterns. Essential for Phase B financial domain problems.

### Reading Room (practice problems)
- **URL**: https://code.kx.com/q/learn/reading/
- Strings: https://code.kx.com/q/learn/reading/strings/
- Fizzbuzz: https://code.kx.com/q/learn/reading/fizzbuzz/
- **Why**: More examples of idiomatic Q solutions.

## K Language References (lower priority)

### K on APL Wiki
- **URL**: https://aplwiki.com/wiki/K
- **Why**: K family overview, version history.

### K Crash Course
- **URL**: https://github.com/kparc/kcc
- **Why**: Concise K tutorial if lower-level K understanding is needed.

### "What about k?" book (ngn/k)
- **URL**: https://xpqz.github.io/kbook/Introduction.html
- **Why**: Open-source K6 reference. Useful for understanding K primitives underlying Q.

## Evaluation Harness

### Q Evaluation Harness
- **URL**: https://github.com/KxSystems/q-evaluation-harness
- **Why**: The test harness for benchmarking. Contains Q-HumanEval (164 problems).

## Suggested Reading Order for Claude Code

1. Q Reference Card (overview of all operators/keywords)
2. Q Syntax Reference (understand evaluation rules)
3. Examples from Python — all 5 pages (see how translations work)
4. Q for Mortals Ch 3 (Lists), Ch 4 (Operators), Ch 6 (Functions)
5. Iteration reference (translating loops to adverbs)
6. Q for Mortals Ch 2 (Data Types), Ch 5 (Dictionaries), Ch 8 (Tables)
7. Programming Idioms
8. Remaining Q for Mortals chapters as needed
