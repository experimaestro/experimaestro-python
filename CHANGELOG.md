## 0.20.0 (2023-02-18)

### Feat

- improvements for dry-run modes to show completed jobs

### Refactor

- more reliable identifier computation

## 0.19.2 (2023-02-16)

### Fix

- better identifier recomputation

## 0.19.1 (2023-02-15)

### Fix

- fix bugs with generate/dry-run modes

## 0.19.0 (2023-02-14)

### Feat

- allow using the old task identifier computation to fix params.json

## 0.18.0 (2023-02-13)

### BREAKING CHANGE

- New identifiers will be different in all cases - use the deprecated command to recompute identifiers for old experiments
- For any task output which is different than the task itself, the identifier will change

### Feat

- **configuration**: re-use computed sub-configuration identifiers

### Fix

- **server**: fix some display bugs in the UI
- **configuration**: fixed more bugs with identifiers
- **configuration**: fixed bugs with identifiers
- **configuration**: serialize the task to recompute exactly the identifier

### Refactor

- removed jsonstreams dependency

## 0.16.0 (2023-02-08)

### Feat

- **server**: web services for experiment server

## 0.15.1 (2023-02-08)

### Fix

- wrong indent
