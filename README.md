# Custom PostgreSQL Vector Store for LlamaIndex

An extended PostgreSQL vector store that provides custom schema control while maintaining full compatibility with LlamaIndex's standard interface.

## <¯ **Complete Solution Overview**

### **Files Created:**
1. **`custom_pg_vector_store.py`** - Main implementation
2. **`example_usage.py`** - Comprehensive examples  
3. **`test_compatibility.py`** - Compatibility verification

### **Key Features Implemented:**

 **Custom Schema Control**
- Add typed columns beyond metadata JSON
- Support for String, Integer, Float, DateTime, Boolean, UUID types
- Configurable nullable, unique, index, and default constraints

 **Foreign Key Relationships**
- Full foreign key support with referential integrity
- Configurable cascade options (CASCADE, RESTRICT, SET NULL)
- Multi-table relational designs

 **Custom Indexes**
- Composite indexes across multiple columns
- Performance optimization for custom queries
- Flexible index configuration

 **Full LlamaIndex Compatibility**
- Inherits from `PGVectorStore` 
- All standard methods work unchanged
- Compatible with `VectorStoreIndex`
- Maintains async/sync operation support

### **Usage Examples:**

```python
# Basic custom columns
schema_config = CustomSchemaConfig(
    custom_columns=[
        create_string_column("author", length=100, nullable=False),
        create_integer_column("page_count"),
        create_datetime_column("published_date")
    ]
)

vector_store = CustomSchemaPGVectorStore(
    connection_string="postgresql://user:pass@localhost:5432/db",
    custom_schema_config=schema_config
)

# Foreign key relationships
schema_config = CustomSchemaConfig(
    custom_columns=[create_uuid_column("author_id")],
    foreign_keys=[
        ForeignKeyConfig("author_id", "authors", "id", on_delete="RESTRICT")
    ]
)
```

### **Benefits Achieved:**
- = **Relational Data**: Proper foreign keys and constraints
- <× **Schema Control**: Full control over table structure
- ¡ **Performance**: Custom indexes for optimal queries
- = **Compatibility**: Works with all existing LlamaIndex code
- =á **Type Safety**: Validated custom data types
- =È **Scalability**: Production-ready PostgreSQL features

The solution successfully bridges the gap between LlamaIndex's convenience and PostgreSQL's relational power, giving you the best of both worlds!

## Getting Started

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Set up your PostgreSQL database with the required extensions:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

3. Use the custom vector store in your application:
   ```python
   from custom_pg_vector_store import CustomSchemaPGVectorStore, CustomSchemaConfig
   
   # Your implementation here
   ```

4. Run the examples:
   ```bash
   uv run python example_usage.py
   ```

5. Run compatibility tests:
   ```bash
   uv run python test_compatibility.py
   ```

## Architecture

The `CustomSchemaPGVectorStore` extends LlamaIndex's `PGVectorStore` using inheritance with schema hooks. This approach:

- **Maintains Compatibility**: All existing LlamaIndex code works unchanged
- **Adds Flexibility**: Custom schema elements are added through configuration
- **Preserves Performance**: Inherits all optimizations (HNSW indexes, hybrid search, etc.)
- **Enables Relations**: Supports proper PostgreSQL foreign keys and constraints

## Configuration Options

### Custom Columns
```python
CustomColumn(
    name="column_name",
    column_type=String(100),  # SQLAlchemy type
    nullable=True,
    unique=False,
    index=False,
    default=None
)
```

### Foreign Keys
```python
ForeignKeyConfig(
    column_name="ref_column",
    referenced_table="other_table",
    referenced_column="id",
    on_delete="CASCADE",  # CASCADE, RESTRICT, SET NULL
    on_update="CASCADE"
)
```

### Custom Indexes
```python
{
    "name": "idx_custom",
    "columns": ["col1", "col2"],
    "unique": False
}
```

This solution provides the control you need for complex relational data while keeping the simplicity and power of LlamaIndex's vector search capabilities.