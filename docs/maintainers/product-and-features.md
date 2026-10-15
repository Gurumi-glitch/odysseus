# Product and feature proposals

Pewds is the Product Owner for Odysseus and has final authority over product direction. Maintainers own technical feasibility, implementation, and safety.

## Which changes require product approval?

**Product Owner approval before implementation:** new features and material changes to product direction or behavior.

**No separate product approval by default:** bug fixes, security fixes, maintenance, tests, CI, documentation, and internal refactors, unless they materially change product behavior or direction.

A security or reliability fix that changes behavior should state the user-visible impact and seek product input when its direction is genuinely ambiguous. Urgent containment need not wait if it preserves existing product intent.

## Product Owner initiated

```text
Product Owner proposal or implementation
    -> technical feasibility
    -> plan and implementation
    -> risk-based review and validation
```

A Product Owner proposal is already product-approved; it is not automatically technically approved or exempt from high-risk review.

## Community or maintainer initiated

```text
Feature proposal
    -> Product Owner accepts / rejects
    -> technical feasibility
    -> plan and implementation
    -> risk-based review and validation
```

Record the decision in an issue, proposal, work item, or PR so future maintainers can discover it. Do not rely on a chat-only decision.

## Decision boundaries

The Product Owner decides **what belongs in Odysseus**. Maintainers decide **how approved work is built safely and maintainably**.

A rejected or out-of-scope feature can be maintained independently in a contributor's fork.

If the Product Owner is unavailable, new product-direction decisions wait unless that authority was explicitly delegated. Urgent security and reliability corrections that do not redefine product direction may continue under normal review rules.
