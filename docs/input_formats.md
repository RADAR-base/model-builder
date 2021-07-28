## Documentation of compatible data format 

### pandas-split

```json
{
    "columns": ["a", "b", "c"],
    "data": [[1, 2, 3], [4, 5, 6]]
}
```

### pandas-records

```json
 '[
    {"a": 1,"b": 2,"c": 3},
    {"a": 4,"b": 5,"c": 6}
]'
```
### tf-instances

```json
'{
    "instances": [
        {"a": "s1", "b": 1, "c": [1, 2, 3]},
        {"a": "s2", "b": 2, "c": [4, 5, 6]},
        {"a": "s3", "b": 3, "c": [7, 8, 9]}
    ]
}'
```

### tf-inputs

```json
'{
    "inputs": {"a": ["s1", "s2", "s3"], "b": [1, 2, 3], "c": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}
}'
```