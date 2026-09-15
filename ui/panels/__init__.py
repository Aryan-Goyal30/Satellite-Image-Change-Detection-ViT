"""Optional Results sections, mounted by result capability.

A panel renders only when the ChangeResult it is given actually carries the data
it describes. Results asks "does this result have direction information?", never
"is this the built-environment domain?", so a domain that gains or loses an
optional capability needs no change here and no domain-name check anywhere.
"""
