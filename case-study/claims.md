# Security Claims

## C1 — General vulnerability claim

Identifier:

```text
vulnerable(C)
```

Reading:

> Contract `C` contains at least one exploitable security vulnerability under
> the analysis scope recorded with the evidence item.

This is the claim most closely aligned with the binary `safe` versus
`vulnerable` classification produced by the Code LLM.

## C2 — Reentrancy exploitability claim

Identifier:

```text
reentrancy-exploitable(C, M)
```

Reading:

> Under attacker and execution model `M`, there exists a finite execution in
> which contract `C` transfers more Ether to the attacker than the attacker is
> entitled to withdraw from the credited balance, because an external call
> permits re-entry before the relevant state update is completed.

This is the primary claim for DivFuzz and model checking.

## Claim relationship

Positive evidence for C2 can support C1:

```text
reentrancy-exploitable(C, M)  =>  vulnerable(C)
```

The converse is invalid:

```text
vulnerable(C)  !=>  reentrancy-exploitable(C, M)
```

Refuting C2 does not refute C1:

```text
not reentrancy-exploitable(C, M)  !=>  not vulnerable(C)
```

Therefore, evidence may be aggregated directly only when it concerns the
same normalized claim and a compatible scope. Claim promotion from C2 to C1
must be explicit and one-way.

## Success predicate for the controlled study

Let:

- `d > 0` be the attacker's credited deposit before the first withdrawal;
- `received` be the Ether received by the attacker during the attack
  transaction.

The controlled reentrancy exploitation succeeds when:

```text
received > d
```

and the excess transfer is caused by repeated entry into `withdraw()` before
the attacker's credited balance is invalidated.
