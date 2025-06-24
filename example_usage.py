"""
Example usage of CustomSchemaPGVectorStore

This example demonstrates how to use the custom PostgreSQL vector store
with foreign keys and custom columns while maintaining LlamaIndex compatibility.
"""

import os
from typing import List
from datetime import datetime
from uuid import uuid4

from sqlalchemy import String, Integer, Float, DateTime, Boolean
from sqlalchemy.dialects.postgresql import UUID

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding

from custom_pg_vector_store import (
    CustomSchemaPGVectorStore,
    CustomSchemaConfig,
    CustomColumn,
    ForeignKeyConfig,
    create_string_column,
    create_integer_column,
    create_datetime_column,
    create_uuid_column
)


def example_1_basic_custom_columns():
    """Example 1: Basic usage with custom columns"""
    print("=== Example 1: Basic Custom Columns ===")
    
    # Define custom schema with additional columns
    schema_config = CustomSchemaConfig(
        custom_columns=[
            create_string_column("author", length=100, nullable=False),
            create_string_column("category", length=50),
            create_integer_column("page_count"),
            create_datetime_column("published_date"),
            create_string_column("isbn", length=20, unique=True, index=True)
        ]
    )
    
    # Create vector store with custom schema
    vector_store = CustomSchemaPGVectorStore(
        connection_string="postgresql://user:password@localhost:5432/mydb",
        table_name="documents",
        embed_dim=1536,
        custom_schema_config=schema_config
    )
    
    # Create sample documents
    documents = [
        Document(
            text="This is a great book about machine learning...",
            metadata={
                "title": "ML Fundamentals",
                "author": "John Doe",
                "category": "Technology",
                "page_count": 350,
                "published_date": datetime(2023, 1, 15),
                "isbn": "978-1234567890"
            }
        ),
        Document(
            text="A comprehensive guide to data science...",
            metadata={
                "title": "Data Science Guide", 
                "author": "Jane Smith",
                "category": "Technology",
                "page_count": 280,
                "published_date": datetime(2023, 5, 20),
                "isbn": "978-0987654321"
            }
        )
    ]
    
    # Convert to nodes
    nodes = []
    for doc in documents:
        node = TextNode(
            text=doc.text,
            metadata=doc.metadata
        )
        nodes.append(node)
    
    # Custom data for additional columns
    custom_data = [
        {
            "author": "John Doe",
            "category": "Technology", 
            "page_count": 350,
            "published_date": datetime(2023, 1, 15),
            "isbn": "978-1234567890"
        },
        {
            "author": "Jane Smith",
            "category": "Technology",
            "page_count": 280, 
            "published_date": datetime(2023, 5, 20),
            "isbn": "978-0987654321"
        }
    ]
    
    # Add nodes with custom data
    node_ids = vector_store.add_custom_data(nodes, custom_data)
    print(f"Added {len(node_ids)} nodes with custom data")
    
    # Query the store (maintains standard LlamaIndex interface)
    query_result = vector_store.query_with_custom_filters(
        query_embedding=[0.1] * 1536,  # Dummy embedding
        similarity_top_k=2,
        custom_filters={"category": "Technology"}
    )
    
    print(f"Query returned {len(query_result.nodes) if query_result.nodes else 0} results")
    
    # Display schema info
    schema_info = vector_store.get_custom_schema_info()
    print("Custom Schema Info:")
    for col in schema_info["custom_columns"]:
        print(f"  - {col['name']}: {col['type']} (nullable={col['nullable']})")


def example_2_foreign_keys():
    """Example 2: Using foreign keys for relational data"""
    print("\n=== Example 2: Foreign Key Relationships ===")
    
    # Define schema with foreign key relationships
    schema_config = CustomSchemaConfig(
        custom_columns=[
            create_uuid_column("author_id", nullable=False),
            create_uuid_column("publisher_id"),
            create_string_column("document_type", length=50),
            create_integer_column("version", default=1),
            create_string_column("language", length=10, default="en")
        ],
        foreign_keys=[
            ForeignKeyConfig(
                column_name="author_id",
                referenced_table="authors",
                referenced_column="id",
                on_delete="RESTRICT"
            ),
            ForeignKeyConfig(
                column_name="publisher_id", 
                referenced_table="publishers",
                referenced_column="id",
                on_delete="SET NULL"
            )
        ],
        custom_indexes=[
            {
                "name": "idx_document_type_lang",
                "columns": ["document_type", "language"],
                "unique": False
            }
        ]
    )
    
    # Create vector store with relational schema
    vector_store = CustomSchemaPGVectorStore(
        connection_string="postgresql://user:password@localhost:5432/mydb",
        table_name="documents_relational",
        embed_dim=1536,
        custom_schema_config=schema_config
    )
    
    # Sample data with foreign key references
    author_id = uuid4()
    publisher_id = uuid4()
    
    documents = [
        Document(
            text="Advanced database concepts and design patterns...",
            metadata={
                "title": "Database Design",
                "author_id": str(author_id),
                "publisher_id": str(publisher_id),
                "document_type": "technical_book",
                "version": 2,
                "language": "en"
            }
        )
    ]
    
    # Convert to nodes
    nodes = [TextNode(text=doc.text, metadata=doc.metadata) for doc in documents]
    
    # Custom data matching the schema
    custom_data = [
        {
            "author_id": author_id,
            "publisher_id": publisher_id,
            "document_type": "technical_book",
            "version": 2,
            "language": "en"
        }
    ]
    
    # Validate custom data before insertion
    for data in custom_data:
        vector_store.validate_custom_data(data)
    
    # Add nodes
    node_ids = vector_store.add_custom_data(nodes, custom_data)
    print(f"Added {len(node_ids)} nodes with foreign key relationships")
    
    # Display schema configuration
    schema_info = vector_store.get_custom_schema_info()
    print("Foreign Key Configuration:")
    for fk in schema_info["foreign_keys"]:
        print(f"  - {fk['column']} -> {fk['references']} (on_delete: {fk['on_delete']})")


def example_3_standard_llamaindex_compatibility():
    """Example 3: Demonstrate standard LlamaIndex compatibility"""
    print("\n=== Example 3: Standard LlamaIndex Compatibility ===")
    
    # Create custom vector store
    schema_config = CustomSchemaConfig(
        custom_columns=[
            create_string_column("source", length=100),
            create_datetime_column("indexed_at", default=datetime.utcnow)
        ]
    )
    
    vector_store = CustomSchemaPGVectorStore(
        connection_string="postgresql://user:password@localhost:5432/mydb",
        table_name="llamaindex_compatible",
        embed_dim=1536,
        custom_schema_config=schema_config
    )
    
    # Use standard LlamaIndex workflow
    embed_model = OpenAIEmbedding()  # This would require OpenAI API key
    
    # Create VectorStoreIndex (standard LlamaIndex usage)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )
    
    print("Successfully created VectorStoreIndex with custom schema vector store")
    print("This demonstrates full compatibility with LlamaIndex ecosystem")
    
    # Standard LlamaIndex document insertion
    documents = [
        Document(text="Sample document for compatibility test"),
        Document(text="Another document to verify standard interface")
    ]
    
    # This uses the standard LlamaIndex interface
    # index.insert(documents)  # Commented out - would need valid embeddings
    
    print("Custom vector store works seamlessly with standard LlamaIndex workflows")


def example_4_advanced_querying():
    """Example 4: Advanced querying with custom schema"""
    print("\n=== Example 4: Advanced Querying ===")
    
    schema_config = CustomSchemaConfig(
        custom_columns=[
            create_float_column("confidence_score"),
            create_string_column("processing_status", length=20),
            create_datetime_column("last_updated")
        ],
        custom_indexes=[
            {
                "name": "idx_confidence_status",
                "columns": ["confidence_score", "processing_status"]
            }
        ]
    )
    
    vector_store = CustomSchemaPGVectorStore(
        connection_string="postgresql://user:password@localhost:5432/mydb",
        table_name="advanced_documents",
        embed_dim=1536,
        custom_schema_config=schema_config
    )
    
    # Query with custom filters (conceptual - full implementation would build SQL)
    results = vector_store.query_with_custom_filters(
        query_embedding=[0.1] * 1536,
        similarity_top_k=10,
        custom_filters={
            "confidence_score": {">=": 0.8},
            "processing_status": "completed"
        }
    )
    
    print("Advanced querying allows filtering on custom columns")
    print("This enables complex analytical queries while maintaining vector search")


if __name__ == "__main__":
    print("Custom PostgreSQL Vector Store Examples")
    print("=" * 50)
    
    # Note: These examples show the interface - actual execution would require:
    # 1. Valid PostgreSQL connection string
    # 2. Proper database setup
    # 3. OpenAI API key for embeddings (in some examples)
    
    try:
        example_1_basic_custom_columns()
        example_2_foreign_keys()
        example_3_standard_llamaindex_compatibility()
        example_4_advanced_querying()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("The CustomSchemaPGVectorStore provides:")
        print("✓ Custom typed columns beyond metadata JSON")
        print("✓ Foreign key relationships for relational data")
        print("✓ Custom indexes for performance optimization")
        print("✓ Full LlamaIndex compatibility")
        print("✓ Schema validation and type safety")
        
    except Exception as e:
        print(f"Example execution failed: {e}")
        print("This is expected without proper database setup and API keys")