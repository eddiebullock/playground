"""
investment data mdoel for individual holdings
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional 
from decimal import Decimal 

@dataclass
class Investment:
    """represents and individual investment holding"""

    id: str 
    portfolio_id: str
    symbol: str # ticker symbol
    name: str
    investment_type: str # stock, bond etc

    # portfolio details 
    quantity: Decimal 
    average_price: Decimal
    current_price: Decimal

    # calculated values
    total_costs: Decimal = Decimal('0')
    current_value: Decimal = Decimal('0')
    unrealized_gain: Decimal = Decimal('0')
    unrealized_gain_loss_percentage: Decimal = Decimal('0')

    # performance tracking 
    last_updated: datetime = datetime.now()
    purchase_date: Optional[datetime] = None

    # Metadata 
    is_active: bool = True
    notes: str = ""

    def __post_init__(self):
        """convert string values to decimal and calculate derived values"""
        # convert string values to decimal

    for field_name in ['quantity', 'average_price', 'current_price', 'total_cost',
                       'current_value', 'unrealized_gain_loss', 'unrealized_gain_loss_percentage']:
        value = getattr(self, field_name)
        if isinstance(value, str):
            setattr(self, field_name, Decimal(value))

    # calculate derived values if not provided
    if self.total_costs == 0:
        self.total_costs = self.quantity * self.average_price

    if self.current_value == 0:
        self.current_value = self.quantity * self.current_price

    if self.unrealized_gain == 0:
        self.unrealized_gain = self.current_value - self.total_costs

    if self.unrealized_gain_loss_percentage == 0:
        