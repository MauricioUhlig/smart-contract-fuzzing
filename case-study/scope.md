# Common Analysis Scope

## Contract scope

- Language target: Solidity `^0.8.20`.
- Target abstraction: a single withdrawal vault with per-address credited balances.
- Both variants expose the same public interface and differ only in the ordering
  of the balance update and the external Ether transfer.
- The two variants must be compiled and analyzed separately because they declare
  the same Solidity contract name.

## Attacker model

The attacker:

- controls a contract capable of receiving Ether;
- can invoke `deposit()` and `withdraw()`;
- can call `withdraw()` again from its receive handler;
- uses a finite re-entry bound;
- has sufficient gas for the bounded execution;
- does not use `delegatecall`, proxy upgrades, `selfdestruct`, miner control, or
  cross-chain behavior.

## Initial-state assumptions

For an attack deposit `d > 0`:

- the attacker is credited with exactly `d`;
- the vault contains at least `2d` immediately before the attacker's first
  withdrawal, so at least one additional transfer of `d` is financially possible;
- the additional vault funds may be supplied by an honest depositor;
- the first controlled experiment uses a maximum of one successful re-entry;
- larger finite bounds may be evaluated as supplementary scenarios.

## External-call semantics

- The low-level Ether call is synchronous.
- The receiver may execute arbitrary bounded code before returning.
- A failed reentrant attempt is caught by the harness and does not automatically
  revert the original Ether transfer.
- Model checking may abstract gas, but must state that sufficient gas is assumed.

## Exploitation criterion

An exploit witness is valid when all the following hold:

1. the attacker initially has credited balance `d`;
2. the attack follows the public contract interface;
3. the attacker receives more than `d` during the attack transaction;
4. the excess is attributable to re-entry before the relevant state update;
5. the trace is reproducible or represented by a valid model-checking counterexample.

## Excluded claims

The case study does not establish:

- complete absence of all smart-contract vulnerabilities;
- security under unbounded gas-sensitive or cross-contract environments;
- equivalence between failure to find a witness and proof of safety;
- correctness of the Solidity compiler or EVM implementation.
