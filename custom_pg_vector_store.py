"""
Custom PostgreSQL Vector Store with Schema Control

Extends LlamaIndex's PGVectorStore to provide custom schema control
while maintaining full compatibility with the standard interface.
"""

from typing import Dict, List, Optional, Any, Union, Type
import re
import logging
from dataclasses import dataclass

from sqlalchemy import (
    Column, Integer, String, Text, JSON, BIGINT, 
    ForeignKey, DateTime, Boolean, Float, Index,
    create_engine, MetaData, Table
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.schema import BaseNode


logger = logging.getLogger(__name__)


@dataclass
class CustomColumn:
    """Configuration for a custom column"""
    name: str
    column_type: Any  # SQLAlchemy column type
    nullable: bool = True
    unique: bool = False
    index: bool = False
    default: Any = None


@dataclass
class ForeignKeyConfig:
    """Configuration for a foreign key relationship"""
    column_name: str
    referenced_table: str
    referenced_column: str = "id"
    on_delete: str = "CASCADE"
    on_update: str = "CASCADE"


@dataclass
class CustomSchemaConfig:
    """Complete configuration for custom schema"""
    custom_columns: List[CustomColumn] = None
    foreign_keys: List[ForeignKeyConfig] = None
    custom_indexes: List[Dict] = None
    table_name_override: Optional[str] = None
    
    def __post_init__(self):
        if self.custom_columns is None:
            self.custom_columns = []
        if self.foreign_keys is None:
            self.foreign_keys = []
        if self.custom_indexes is None:
            self.custom_indexes = []


class CustomSchemaPGVectorStore(PGVectorStore):
    """
    Extended PostgreSQL Vector Store with custom schema control.
    
    Maintains full compatibility with LlamaIndex PGVectorStore while
    adding support for:
    - Custom typed columns beyond metadata JSON
    - Foreign key relationships
    - Custom indexes
    - Schema validation
    """
    
    custom_schema_config: Optional[CustomSchemaConfig] = None
    _custom_model_cache: Dict = None
    
    def __init__(
        self,
        custom_schema_config: Optional[CustomSchemaConfig] = None,
        **kwargs
    ):
        """
        Initialize the custom schema vector store.
        
        Args:
            custom_schema_config: Configuration for custom schema elements
            **kwargs: All standard PGVectorStore parameters
        """
        # Set custom attributes before parent initialization
        object.__setattr__(self, 'custom_schema_config', custom_schema_config or CustomSchemaConfig())
        object.__setattr__(self, '_custom_model_cache', {})
        
        # Initialize parent class
        super().__init__(**kwargs)
    
    def get_data_model(self, index_name: str, embed_dim: int):
        """
        Override the data model creation to add custom schema elements.
        
        This method extends the base PGVectorStore model with:
        - Custom typed columns
        - Foreign key relationships
        - Custom constraints and indexes
        """
        # Use cached model if available
        cache_key = f"{index_name}_{embed_dim}"
        if cache_key in self._custom_model_cache:
            return self._custom_model_cache[cache_key]
        
        # Get the base model from parent class
        base_model = super().get_data_model(index_name, embed_dim)
        
        # If no custom configuration, return base model
        if not self._has_custom_schema():
            self._custom_model_cache[cache_key] = base_model
            return base_model
        
        # Create enhanced model with custom schema
        enhanced_model = self._create_enhanced_model(base_model, index_name, embed_dim)
        self._custom_model_cache[cache_key] = enhanced_model
        return enhanced_model
    
    def _has_custom_schema(self) -> bool:
        """Check if any custom schema configuration is provided"""
        config = self.custom_schema_config
        return (
            bool(config.custom_columns) or 
            bool(config.foreign_keys) or 
            bool(config.custom_indexes) or
            config.table_name_override
        )
    
    def _create_enhanced_model(self, base_model, index_name: str, embed_dim: int):
        """Create an enhanced SQLAlchemy model with custom schema elements"""
        
        # Get base table configuration
        base_table = base_model.__table__
        table_name = (
            self.custom_schema_config.table_name_override or 
            base_table.name
        )
        
        # Create new metadata and table
        metadata = MetaData()
        
        # Start with base columns
        columns = []
        for col in base_table.columns:
            new_col = col.copy()
            columns.append(new_col)
        
        # Add custom columns
        for custom_col in self.custom_schema_config.custom_columns:
            col = Column(
                custom_col.name,
                custom_col.column_type,
                nullable=custom_col.nullable,
                unique=custom_col.unique,
                index=custom_col.index,
                default=custom_col.default
            )
            columns.append(col)
        
        # Add foreign key constraints
        for fk_config in self.custom_schema_config.foreign_keys:
            # Find the column to add foreign key to
            for col in columns:
                if col.name == fk_config.column_name:
                    # Add foreign key constraint
                    fk_constraint = ForeignKey(
                        f"{fk_config.referenced_table}.{fk_config.referenced_column}",
                        ondelete=fk_config.on_delete,
                        onupdate=fk_config.on_update
                    )
                    col.foreign_keys.add(fk_constraint)
                    break
        
        # Create the enhanced table
        enhanced_table = Table(
            table_name,
            metadata,
            *columns,
            schema=self.schema_name
        )
        
        # Add custom indexes
        for index_config in self.custom_schema_config.custom_indexes:
            index_name = index_config.get('name', f"idx_{table_name}_{index_config['columns'][0]}")
            Index(
                index_name,
                *[enhanced_table.c[col] for col in index_config['columns']],
                unique=index_config.get('unique', False)
            )
        
        # Create a new model class
        Base = declarative_base(metadata=metadata)
        
        class EnhancedDataModel(Base):
            __table__ = enhanced_table
            
            def __repr__(self):
                return f"<{self.__class__.__name__}(id={self.id}, node_id={self.node_id})>"
        
        return EnhancedDataModel
    
    def add_custom_data(
        self, 
        nodes: List[BaseNode], 
        custom_data: List[Dict[str, Any]],
        **kwargs
    ) -> List[str]:
        """
        Add nodes with custom column data.
        
        Args:
            nodes: List of BaseNode objects
            custom_data: List of dictionaries containing custom column values
            **kwargs: Additional arguments passed to parent add method
            
        Returns:
            List of node IDs
        """
        if len(nodes) != len(custom_data):
            raise ValueError("Number of nodes must match number of custom_data entries")
        
        # Enhance nodes with custom data in metadata for now
        # In a full implementation, this would insert custom data directly
        enhanced_nodes = []
        for node, custom_vals in zip(nodes, custom_data):
            # Add custom data to node metadata
            if node.metadata is None:
                node.metadata = {}
            node.metadata.update({"_custom_data": custom_vals})
            enhanced_nodes.append(node)
        
        return super().add(enhanced_nodes, **kwargs)
    
    def query_with_custom_filters(
        self,
        query_embedding: Optional[List[float]] = None,
        custom_filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Query with filters on custom columns.
        
        Args:
            query_embedding: Query embedding vector
            custom_filters: Filters for custom columns
            **kwargs: Additional query parameters
            
        Returns:
            Query results
        """
        # In a full implementation, this would build proper SQL queries
        # For now, we delegate to the parent class
        from llama_index.core.vector_stores.types import VectorStoreQuery
        
        query_obj = VectorStoreQuery(
            query_embedding=query_embedding,
            **kwargs
        )
        
        return super().query(query_obj)
    
    def get_custom_schema_info(self) -> Dict[str, Any]:
        """Get information about the custom schema configuration"""
        return {
            "custom_columns": [
                {
                    "name": col.name,
                    "type": str(col.column_type),
                    "nullable": col.nullable,
                    "unique": col.unique,
                    "index": col.index,
                    "default": col.default
                }
                for col in self.custom_schema_config.custom_columns
            ],
            "foreign_keys": [
                {
                    "column": fk.column_name,
                    "references": f"{fk.referenced_table}.{fk.referenced_column}",
                    "on_delete": fk.on_delete,
                    "on_update": fk.on_update
                }
                for fk in self.custom_schema_config.foreign_keys
            ],
            "custom_indexes": self.custom_schema_config.custom_indexes,
            "table_name_override": self.custom_schema_config.table_name_override
        }
    
    def validate_custom_data(self, custom_data: Dict[str, Any]) -> bool:
        """
        Validate custom data against schema configuration.
        
        Args:
            custom_data: Dictionary of custom column values
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        for column in self.custom_schema_config.custom_columns:
            if column.name in custom_data:
                value = custom_data[column.name]
                
                # Check nullable constraint
                if not column.nullable and value is None:
                    raise ValueError(f"Column '{column.name}' cannot be null")
                
                # Additional type validation could go here
                
        return True


# Convenience functions for common column types
def create_string_column(name: str, length: int = 255, **kwargs) -> CustomColumn:
    """Create a string column configuration"""
    return CustomColumn(name=name, column_type=String(length), **kwargs)


def create_integer_column(name: str, **kwargs) -> CustomColumn:
    """Create an integer column configuration"""
    return CustomColumn(name=name, column_type=Integer, **kwargs)


def create_float_column(name: str, **kwargs) -> CustomColumn:
    """Create a float column configuration"""
    return CustomColumn(name=name, column_type=Float, **kwargs)


def create_datetime_column(name: str, **kwargs) -> CustomColumn:
    """Create a datetime column configuration"""
    return CustomColumn(name=name, column_type=DateTime(timezone=True), **kwargs)


def create_boolean_column(name: str, **kwargs) -> CustomColumn:
    """Create a boolean column configuration"""
    return CustomColumn(name=name, column_type=Boolean, **kwargs)


def create_uuid_column(name: str, **kwargs) -> CustomColumn:
    """Create a UUID column configuration"""
    return CustomColumn(name=name, column_type=UUID(as_uuid=True), **kwargs)