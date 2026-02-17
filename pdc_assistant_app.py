"""
⚡ PDC Assistant (Protection & Device Coordination Assistant)
==============================================================
Senior Power System Protection Engineer Tool
IEEE 242 & IEC 60255 Standards Compliant

Author: Protection Engineering Team
Version: 1.0.0
Python: 3.10+
Framework: Streamlit

Description:
    A comprehensive web application for power system protection coordination.
    Calculates relay settings, validates coordination, and ensures compliance
    with IEEE 242 and IEC 60255 standards.

Features:
    - Equipment modeling (Transformer, Generator, UPS, Cable, Motor)
    - Auto-calculation of protection settings
    - IEC 60255 curve equations (SI, VI, EI, LTI)
    - CTI validation (Coordination Time Interval)
    - IEEE 242 recommended practices
    - Export to CSV
    - Advanced mode with detailed calculations
"""

import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum
import io
import math

# ============================================================================
# CONSTANTS AND ENUMS
# ============================================================================

class EquipmentType(Enum):
    """Equipment types supported by PDC Assistant"""
    TRANSFORMER = "Transformer"
    GENERATOR = "Generator"
    UPS = "UPS"
    CABLE = "Cable"
    MOTOR = "Motor"
    LOAD = "Load"

class CurveType(Enum):
    """IEC 60255 Curve Types"""
    STANDARD_INVERSE = "Standard Inverse (SI)"
    VERY_INVERSE = "Very Inverse (VI)"
    EXTREMELY_INVERSE = "Extremely Inverse (EI)"
    LONG_TIME_INVERSE = "Long Time Inverse (LTI)"

class CoordinationStatus(Enum):
    """Coordination validation status"""
    OK = "✅ OK"
    MARGINAL = "⚠️ Marginal"
    FAILED = "❌ Failed"

# IEC 60255 Curve Constants
IEC_CURVE_CONSTANTS = {
    CurveType.STANDARD_INVERSE: {"k": 0.14, "alpha": 0.02},
    CurveType.VERY_INVERSE: {"k": 13.5, "alpha": 1.0},
    CurveType.EXTREMELY_INVERSE: {"k": 80.0, "alpha": 2.0},
    CurveType.LONG_TIME_INVERSE: {"k": 120.0, "alpha": 1.0}
}

# IEEE 242 Recommended Settings
IEEE_242_DEFAULTS = {
    "transformer_51p_multiplier": 1.25,
    "transformer_50p_multiplier": 10.0,
    "transformer_51n_percent": 15.0,
    "generator_51p_multiplier": 1.15,
    "generator_reverse_power_percent": 5.0,
    "ups_instantaneous_multiplier": 2.5,
    "cable_pickup_safety_factor": 0.95,
    "minimum_cti": 0.2,  # seconds
    "recommended_cti": 0.3  # seconds
}

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Equipment:
    """Base class for all equipment"""
    name: str
    equipment_type: EquipmentType
    voltage_kv: float
    kva: float = 0.0
    rated_current: float = 0.0
    
    def calculate_full_load_current(self) -> float:
        """
        Calculate full load current using:
        I = S / (√3 × V)
        
        Returns:
            float: Full load current in Amperes
        """
        if self.kva > 0 and self.voltage_kv > 0:
            return self.kva * 1000 / (math.sqrt(3) * self.voltage_kv * 1000)
        return self.rated_current

@dataclass
class Transformer(Equipment):
    """Transformer equipment data"""
    impedance_percent: float = 5.5
    inrush_multiplier: float = 8.0
    secondary_voltage_kv: float = 0.415
    vector_group: str = "Dyn11"
    
    def calculate_short_circuit_current(self) -> float:
        """
        Calculate transformer short circuit current:
        Isc = I_fl / (Z_pu)
        
        Returns:
            float: Short circuit current in Amperes
        """
        i_fl = self.calculate_full_load_current()
        z_pu = self.impedance_percent / 100.0
        return i_fl / z_pu if z_pu > 0 else 0.0
    
    def calculate_inrush_current(self) -> float:
        """Calculate transformer inrush current"""
        i_fl = self.calculate_full_load_current()
        return i_fl * self.inrush_multiplier

@dataclass
class Generator(Equipment):
    """Generator equipment data"""
    xd_subtransient: float = 0.15  # X"d in per unit
    power_factor: float = 0.8
    
    def calculate_subtransient_current(self) -> float:
        """
        Calculate generator subtransient fault current:
        I" = I_rated / X"d
        
        Returns:
            float: Subtransient current in Amperes
        """
        i_rated = self.calculate_full_load_current()
        return i_rated / self.xd_subtransient if self.xd_subtransient > 0 else 0.0

@dataclass
class UPS(Equipment):
    """UPS equipment data"""
    fault_current_multiplier: float = 2.0
    fault_clearing_duration_ms: float = 100.0
    battery_backup_minutes: float = 30.0
    
    def calculate_fault_current(self) -> float:
        """Calculate UPS maximum fault current contribution"""
        i_rated = self.calculate_full_load_current()
        return i_rated * self.fault_current_multiplier

@dataclass
class Cable(Equipment):
    """Cable equipment data"""
    length_m: float = 100.0
    resistance_mohm_per_m: float = 0.124
    reactance_mohm_per_m: float = 0.08
    size_mm2: float = 185.0
    k_constant: float = 143.0  # Copper: 143, Aluminum: 94
    ampacity: float = 0.0
    
    def calculate_resistance(self) -> float:
        """Calculate total cable resistance in ohms"""
        return (self.resistance_mohm_per_m * self.length_m) / 1000.0
    
    def calculate_reactance(self) -> float:
        """Calculate total cable reactance in ohms"""
        return (self.reactance_mohm_per_m * self.length_m) / 1000.0
    
    def calculate_impedance(self) -> float:
        """Calculate cable impedance magnitude"""
        r = self.calculate_resistance()
        x = self.calculate_reactance()
        return math.sqrt(r**2 + x**2)
    
    def calculate_thermal_limit(self, duration_sec: float) -> float:
        """
        Calculate cable thermal withstand:
        I²t ≤ k²S²
        
        Args:
            duration_sec: Fault duration in seconds
            
        Returns:
            float: Maximum current for given duration
        """
        if duration_sec > 0:
            return (self.k_constant * self.size_mm2) / math.sqrt(duration_sec)
        return 0.0

@dataclass
class Motor(Equipment):
    """Motor equipment data"""
    locked_rotor_multiplier: float = 6.0
    service_factor: float = 1.15
    efficiency: float = 0.95
    
    def calculate_locked_rotor_current(self) -> float:
        """Calculate motor locked rotor (starting) current"""
        i_fl = self.calculate_full_load_current()
        return i_fl * self.locked_rotor_multiplier

@dataclass
class RelaySettings:
    """Relay protection settings"""
    equipment_name: str
    
    # Overcurrent Phase (51P)
    phase_pickup_current: float = 0.0
    phase_curve_type: CurveType = CurveType.STANDARD_INVERSE
    phase_tms: float = 0.1  # Time Multiplier Setting
    
    # Instantaneous Phase (50P)
    instantaneous_enabled: bool = False
    instantaneous_pickup: float = 0.0
    
    # Earth Fault (51N)
    earth_fault_pickup: float = 0.0
    earth_fault_curve_type: CurveType = CurveType.STANDARD_INVERSE
    earth_fault_tms: float = 0.1
    
    # Special protections
    second_harmonic_blocking: bool = False
    reverse_power_enabled: bool = False
    reverse_power_percent: float = 5.0
    
    # Custom override
    use_custom: bool = False

# ============================================================================
# CALCULATION ENGINE
# ============================================================================

class ProtectionCalculator:
    """Core calculation engine for protection coordination"""
    
    @staticmethod
    def calculate_iec_curve_time(
        current: float,
        pickup: float,
        tms: float,
        curve_type: CurveType
    ) -> float:
        """
        Calculate relay operating time using IEC 60255 equation:
        t = (k × TMS) / ((I / I_pickup)^α - 1)
        
        Args:
            current: Fault current in Amperes
            pickup: Relay pickup current in Amperes
            tms: Time Multiplier Setting
            curve_type: IEC curve type
            
        Returns:
            float: Operating time in seconds
        """
        if pickup <= 0 or current <= pickup:
            return float('inf')
        
        constants = IEC_CURVE_CONSTANTS[curve_type]
        k = constants["k"]
        alpha = constants["alpha"]
        
        ratio = current / pickup
        
        try:
            time = (k * tms) / (ratio**alpha - 1)
            return max(0.0, time)
        except (ZeroDivisionError, ValueError):
            return float('inf')
    
    @staticmethod
    def calculate_cti(
        upstream_time: float,
        downstream_time: float
    ) -> float:
        """
        Calculate Coordination Time Interval (CTI):
        CTI = T_upstream - T_downstream
        
        Args:
            upstream_time: Upstream relay operating time (seconds)
            downstream_time: Downstream relay operating time (seconds)
            
        Returns:
            float: CTI in seconds
        """
        return upstream_time - downstream_time
    
    @staticmethod
    def validate_cable_thermal(
        cable: Cable,
        fault_current: float,
        clearing_time: float
    ) -> Tuple[bool, float]:
        """
        Validate cable thermal withstand:
        I²t_actual ≤ I²t_cable
        
        Args:
            cable: Cable object
            fault_current: Fault current through cable
            clearing_time: Relay clearing time
            
        Returns:
            tuple: (is_valid, margin_ratio)
        """
        i2t_actual = (fault_current ** 2) * clearing_time
        i_thermal = cable.calculate_thermal_limit(clearing_time)
        i2t_limit = (i_thermal ** 2) * clearing_time
        
        if i2t_limit > 0:
            margin = i2t_limit / i2t_actual
            return (margin >= 1.0, margin)
        return (False, 0.0)

# ============================================================================
# RECOMMENDATION ENGINE
# ============================================================================

class RecommendationEngine:
    """IEEE 242 compliant relay settings recommendation engine"""
    
    @staticmethod
    def recommend_transformer_settings(transformer: Transformer) -> RelaySettings:
        """
        Recommend transformer protection settings per IEEE 242
        
        Primary Side (51P, 50P, 51N):
        - 51P pickup = 1.25 × FLA
        - 50P = 10 × FLA (or disabled if coordination required)
        - 51N = 15% of FLA
        - 2nd harmonic blocking enabled
        
        Args:
            transformer: Transformer object
            
        Returns:
            RelaySettings: Recommended settings
        """
        i_fl = transformer.calculate_full_load_current()
        
        settings = RelaySettings(
            equipment_name=transformer.name,
            phase_pickup_current=i_fl * IEEE_242_DEFAULTS["transformer_51p_multiplier"],
            phase_curve_type=CurveType.STANDARD_INVERSE,
            phase_tms=0.1,
            instantaneous_enabled=True,
            instantaneous_pickup=i_fl * IEEE_242_DEFAULTS["transformer_50p_multiplier"],
            earth_fault_pickup=i_fl * (IEEE_242_DEFAULTS["transformer_51n_percent"] / 100.0),
            earth_fault_curve_type=CurveType.STANDARD_INVERSE,
            earth_fault_tms=0.1,
            second_harmonic_blocking=True
        )
        
        return settings
    
    @staticmethod
    def recommend_generator_settings(generator: Generator) -> RelaySettings:
        """
        Recommend generator protection settings per IEEE 242
        
        Protection (51P, 32, 50):
        - 51P pickup = 1.15 × FLA
        - Reverse power = 5% rated
        - 50P < I_subtransient
        
        Args:
            generator: Generator object
            
        Returns:
            RelaySettings: Recommended settings
        """
        i_fl = generator.calculate_full_load_current()
        i_subtrans = generator.calculate_subtransient_current()
        
        settings = RelaySettings(
            equipment_name=generator.name,
            phase_pickup_current=i_fl * IEEE_242_DEFAULTS["generator_51p_multiplier"],
            phase_curve_type=CurveType.VERY_INVERSE,
            phase_tms=0.15,
            instantaneous_enabled=True,
            instantaneous_pickup=min(i_subtrans * 0.8, i_fl * 8.0),
            earth_fault_pickup=i_fl * 0.1,
            earth_fault_curve_type=CurveType.STANDARD_INVERSE,
            earth_fault_tms=0.1,
            reverse_power_enabled=True,
            reverse_power_percent=IEEE_242_DEFAULTS["generator_reverse_power_percent"]
        )
        
        return settings
    
    @staticmethod
    def recommend_ups_settings(ups: UPS) -> RelaySettings:
        """
        Recommend UPS protection settings
        
        Protection:
        - 51P pickup = 1.1 × rated
        - Instantaneous disabled or > 2.5 × rated
        
        Args:
            ups: UPS object
            
        Returns:
            RelaySettings: Recommended settings
        """
        i_rated = ups.calculate_full_load_current()
        
        settings = RelaySettings(
            equipment_name=ups.name,
            phase_pickup_current=i_rated * 1.1,
            phase_curve_type=CurveType.STANDARD_INVERSE,
            phase_tms=0.05,
            instantaneous_enabled=False,  # Typically disabled for UPS
            instantaneous_pickup=i_rated * IEEE_242_DEFAULTS["ups_instantaneous_multiplier"],
            earth_fault_pickup=i_rated * 0.2,
            earth_fault_curve_type=CurveType.STANDARD_INVERSE,
            earth_fault_tms=0.05
        )
        
        return settings
    
    @staticmethod
    def recommend_cable_settings(
        cable: Cable,
        downstream_load_current: float
    ) -> RelaySettings:
        """
        Recommend cable protection settings
        
        Protection:
        - Pickup ≤ cable ampacity
        - I²t protection
        
        Args:
            cable: Cable object
            downstream_load_current: Load current through cable
            
        Returns:
            RelaySettings: Recommended settings
        """
        # Use cable ampacity or downstream load
        pickup = min(
            cable.ampacity * IEEE_242_DEFAULTS["cable_pickup_safety_factor"],
            downstream_load_current * 1.25
        ) if cable.ampacity > 0 else downstream_load_current * 1.25
        
        settings = RelaySettings(
            equipment_name=cable.name,
            phase_pickup_current=pickup,
            phase_curve_type=CurveType.STANDARD_INVERSE,
            phase_tms=0.1,
            instantaneous_enabled=True,
            instantaneous_pickup=pickup * 10.0,
            earth_fault_pickup=pickup * 0.2,
            earth_fault_curve_type=CurveType.STANDARD_INVERSE,
            earth_fault_tms=0.1
        )
        
        return settings
    
    @staticmethod
    def recommend_motor_settings(motor: Motor) -> RelaySettings:
        """
        Recommend motor protection settings
        
        Protection (49, 51):
        - Thermal overload = 1.15 × FLA
        - Locked rotor consideration
        
        Args:
            motor: Motor object
            
        Returns:
            RelaySettings: Recommended settings
        """
        i_fl = motor.calculate_full_load_current()
        i_lr = motor.calculate_locked_rotor_current()
        
        settings = RelaySettings(
            equipment_name=motor.name,
            phase_pickup_current=i_fl * motor.service_factor,
            phase_curve_type=CurveType.STANDARD_INVERSE,
            phase_tms=0.2,
            instantaneous_enabled=True,
            instantaneous_pickup=i_lr * 1.1,  # Just above locked rotor
            earth_fault_pickup=i_fl * 0.2,
            earth_fault_curve_type=CurveType.STANDARD_INVERSE,
            earth_fault_tms=0.1
        )
        
        return settings

# ============================================================================
# VALIDATION ENGINE
# ============================================================================

class CoordinationValidator:
    """Validates protection coordination between devices"""
    
    @staticmethod
    def validate_coordination(
        upstream_settings: RelaySettings,
        downstream_settings: RelaySettings,
        fault_current: float,
        equipment_data: Optional[Equipment] = None
    ) -> Dict:
        """
        Comprehensive coordination validation
        
        Checks:
        1. Pickup current selectivity
        2. CTI at fault current
        3. Equipment damage curve compliance
        4. Instantaneous coordination
        
        Args:
            upstream_settings: Upstream relay settings
            downstream_settings: Downstream relay settings
            fault_current: Fault current for CTI check
            equipment_data: Optional equipment for damage curve check
            
        Returns:
            dict: Validation results with status and details
        """
        calc = ProtectionCalculator()
        results = {
            "pickup_ok": False,
            "cti_ok": False,
            "cti_value": 0.0,
            "instantaneous_ok": False,
            "damage_curve_ok": True,
            "overall_status": CoordinationStatus.FAILED,
            "messages": []
        }
        
        # Check 1: Pickup selectivity
        if upstream_settings.phase_pickup_current > downstream_settings.phase_pickup_current:
            results["pickup_ok"] = True
            results["messages"].append("✅ Pickup selectivity: OK")
        else:
            results["messages"].append("❌ Pickup selectivity: Failed - Upstream pickup must be > downstream")
        
        # Check 2: CTI validation
        t_downstream = calc.calculate_iec_curve_time(
            fault_current,
            downstream_settings.phase_pickup_current,
            downstream_settings.phase_tms,
            downstream_settings.phase_curve_type
        )
        
        t_upstream = calc.calculate_iec_curve_time(
            fault_current,
            upstream_settings.phase_pickup_current,
            upstream_settings.phase_tms,
            upstream_settings.phase_curve_type
        )
        
        cti = calc.calculate_cti(t_upstream, t_downstream)
        results["cti_value"] = cti
        
        if cti >= IEEE_242_DEFAULTS["recommended_cti"]:
            results["cti_ok"] = True
            results["messages"].append(f"✅ CTI = {cti:.3f}s (Recommended ≥ 0.3s): Excellent")
        elif cti >= IEEE_242_DEFAULTS["minimum_cti"]:
            results["cti_ok"] = True
            results["messages"].append(f"⚠️ CTI = {cti:.3f}s (Minimum ≥ 0.2s): Acceptable")
        else:
            results["messages"].append(f"❌ CTI = {cti:.3f}s: Failed - Must be ≥ 0.2s")
        
        # Check 3: Instantaneous coordination
        if upstream_settings.instantaneous_enabled and downstream_settings.instantaneous_enabled:
            if upstream_settings.instantaneous_pickup > downstream_settings.instantaneous_pickup * 1.25:
                results["instantaneous_ok"] = True
                results["messages"].append("✅ Instantaneous coordination: OK")
            else:
                results["messages"].append("⚠️ Instantaneous coordination: Marginal - Consider 25% margin")
        else:
            results["instantaneous_ok"] = True
            results["messages"].append("ℹ️ Instantaneous: One or both disabled")
        
        # Check 4: Equipment damage curve (if applicable)
        if equipment_data and isinstance(equipment_data, (Transformer, Generator)):
            if isinstance(equipment_data, Transformer):
                # Check inrush vs instantaneous
                i_inrush = equipment_data.calculate_inrush_current()
                if upstream_settings.instantaneous_enabled:
                    if upstream_settings.instantaneous_pickup > i_inrush * 1.5:
                        results["messages"].append(f"✅ Transformer inrush protection: OK ({i_inrush:.0f}A)")
                    else:
                        results["damage_curve_ok"] = False
                        results["messages"].append(f"❌ Instantaneous may trip on inrush ({i_inrush:.0f}A)")
            
            elif isinstance(equipment_data, Generator):
                # Check subtransient tolerance
                i_subtrans = equipment_data.calculate_subtransient_current()
                if upstream_settings.instantaneous_pickup < i_subtrans:
                    results["messages"].append(f"✅ Generator fault tolerance: OK ({i_subtrans:.0f}A)")
                else:
                    results["damage_curve_ok"] = False
                    results["messages"].append(f"⚠️ Instantaneous > subtransient current ({i_subtrans:.0f}A)")
        
        # Overall status determination
        if results["pickup_ok"] and results["cti_ok"] and results["damage_curve_ok"]:
            if cti >= IEEE_242_DEFAULTS["recommended_cti"] and results["instantaneous_ok"]:
                results["overall_status"] = CoordinationStatus.OK
            else:
                results["overall_status"] = CoordinationStatus.MARGINAL
        else:
            results["overall_status"] = CoordinationStatus.FAILED
        
        return results

# ============================================================================
# STREAMLIT UI APPLICATION
# ============================================================================

def init_session_state():
    """Initialize Streamlit session state variables"""
    if 'equipment_list' not in st.session_state:
        st.session_state.equipment_list = []
    if 'relay_settings' not in st.session_state:
        st.session_state.relay_settings = {}
    if 'project_name' not in st.session_state:
        st.session_state.project_name = "New Protection Coordination Study"
    if 'advanced_mode' not in st.session_state:
        st.session_state.advanced_mode = False

def sidebar_setup():
    """Sidebar for project configuration and navigation"""
    st.sidebar.title("⚡ PDC Assistant")
    st.sidebar.markdown("**Protection & Device Coordination**")
    st.sidebar.markdown("---")
    
    # Project setup
    st.sidebar.subheader("📋 Project Setup")
    st.session_state.project_name = st.sidebar.text_input(
        "Project Name",
        value=st.session_state.project_name
    )
    
    # Advanced mode toggle
    st.session_state.advanced_mode = st.sidebar.checkbox(
        "🔬 Advanced Mode",
        value=st.session_state.advanced_mode,
        help="Show detailed calculations and equations"
    )
    
    st.sidebar.markdown("---")
    
    # Quick stats
    st.sidebar.subheader("📊 Current Study")
    st.sidebar.metric("Equipment Count", len(st.session_state.equipment_list))
    st.sidebar.metric("Relay Settings", len(st.session_state.relay_settings))
    
    st.sidebar.markdown("---")
    
    # Actions
    st.sidebar.subheader("🔧 Actions")
    if st.sidebar.button("🗑️ Clear All Data", use_container_width=True):
        st.session_state.equipment_list = []
        st.session_state.relay_settings = {}
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Standards Compliance:**")
    st.sidebar.markdown("✓ IEEE 242-2001")
    st.sidebar.markdown("✓ IEC 60255")
    st.sidebar.markdown("✓ IEEE C37.112")

def equipment_data_tab():
    """Tab for equipment data entry"""
    st.header("⚙️ Equipment Data Entry")
    
    # Equipment type selector
    col1, col2 = st.columns([2, 1])
    
    with col1:
        equipment_type = st.selectbox(
            "Select Equipment Type",
            options=[e.value for e in EquipmentType],
            key="equipment_type_selector"
        )
    
    with col2:
        if st.button("➕ Add Equipment", type="primary", use_container_width=True):
            st.session_state.show_add_form = True
    
    # Add equipment form
    if st.session_state.get('show_add_form', False):
        with st.expander("📝 New Equipment Entry", expanded=True):
            add_equipment_form(equipment_type)
    
    # Display existing equipment
    if st.session_state.equipment_list:
        st.markdown("---")
        st.subheader("📋 Existing Equipment")
        
        for idx, equipment in enumerate(st.session_state.equipment_list):
            with st.expander(f"🔌 {equipment.name} ({equipment.equipment_type.value})"):
                display_equipment_details(equipment, idx)

def add_equipment_form(equipment_type_str: str):
    """Form for adding new equipment"""
    equipment_type = EquipmentType(equipment_type_str)
    
    # Common fields
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Equipment Name", key="eq_name")
        voltage_kv = st.number_input("Voltage (kV)", min_value=0.0, value=11.0, step=0.1, key="eq_voltage")
    
    with col2:
        kva = st.number_input("Rating (kVA)", min_value=0.0, value=1000.0, step=100.0, key="eq_kva")
    
    # Type-specific fields
    if equipment_type == EquipmentType.TRANSFORMER:
        col3, col4 = st.columns(2)
        with col3:
            impedance = st.number_input("Impedance (%)", min_value=0.0, value=5.5, step=0.1, key="tx_z")
            inrush = st.number_input("Inrush Multiplier", min_value=1.0, value=8.0, step=0.5, key="tx_inrush")
        with col4:
            sec_voltage = st.number_input("Secondary Voltage (kV)", min_value=0.0, value=0.415, step=0.001, key="tx_sec_v")
            vector = st.selectbox("Vector Group", ["Dyn11", "Dyn1", "Yyn0", "Dd0"], key="tx_vector")
    
    elif equipment_type == EquipmentType.GENERATOR:
        col3, col4 = st.columns(2)
        with col3:
            xd = st.number_input("X\"d (pu)", min_value=0.0, value=0.15, step=0.01, key="gen_xd")
        with col4:
            pf = st.number_input("Power Factor", min_value=0.0, max_value=1.0, value=0.8, step=0.05, key="gen_pf")
    
    elif equipment_type == EquipmentType.UPS:
        col3, col4 = st.columns(2)
        with col3:
            fault_mult = st.number_input("Fault Current Multiplier", min_value=1.0, value=2.0, step=0.1, key="ups_fault")
        with col4:
            backup = st.number_input("Backup Time (min)", min_value=0.0, value=30.0, step=5.0, key="ups_backup")
    
    elif equipment_type == EquipmentType.CABLE:
        col3, col4 = st.columns(2)
        with col3:
            length = st.number_input("Length (m)", min_value=0.0, value=100.0, step=10.0, key="cable_len")
            r_mohm = st.number_input("R (mΩ/m)", min_value=0.0, value=0.124, step=0.001, format="%.3f", key="cable_r")
            size = st.number_input("Size (mm²)", min_value=0.0, value=185.0, step=10.0, key="cable_size")
        with col4:
            x_mohm = st.number_input("X (mΩ/m)", min_value=0.0, value=0.08, step=0.001, format="%.3f", key="cable_x")
            k_const = st.number_input("k constant", min_value=0.0, value=143.0, step=1.0, help="143 for Cu, 94 for Al", key="cable_k")
            ampacity = st.number_input("Ampacity (A)", min_value=0.0, value=400.0, step=10.0, key="cable_amp")
    
    elif equipment_type == EquipmentType.MOTOR:
        col3, col4 = st.columns(2)
        with col3:
            lr_mult = st.number_input("Locked Rotor Multiplier", min_value=1.0, value=6.0, step=0.5, key="motor_lr")
            sf = st.number_input("Service Factor", min_value=1.0, value=1.15, step=0.05, key="motor_sf")
        with col4:
            eff = st.number_input("Efficiency", min_value=0.0, max_value=1.0, value=0.95, step=0.01, key="motor_eff")
    
    # Add button
    col_add1, col_add2 = st.columns([3, 1])
    with col_add2:
        if st.button("✅ Add", type="primary", use_container_width=True):
            # Create equipment object
            if equipment_type == EquipmentType.TRANSFORMER:
                equipment = Transformer(
                    name=name, equipment_type=equipment_type, voltage_kv=voltage_kv, kva=kva,
                    impedance_percent=impedance, inrush_multiplier=inrush,
                    secondary_voltage_kv=sec_voltage, vector_group=vector
                )
            elif equipment_type == EquipmentType.GENERATOR:
                equipment = Generator(
                    name=name, equipment_type=equipment_type, voltage_kv=voltage_kv, kva=kva,
                    xd_subtransient=xd, power_factor=pf
                )
            elif equipment_type == EquipmentType.UPS:
                equipment = UPS(
                    name=name, equipment_type=equipment_type, voltage_kv=voltage_kv, kva=kva,
                    fault_current_multiplier=fault_mult, battery_backup_minutes=backup
                )
            elif equipment_type == EquipmentType.CABLE:
                equipment = Cable(
                    name=name, equipment_type=equipment_type, voltage_kv=voltage_kv,
                    length_m=length, resistance_mohm_per_m=r_mohm, reactance_mohm_per_m=x_mohm,
                    size_mm2=size, k_constant=k_const, ampacity=ampacity
                )
            elif equipment_type == EquipmentType.MOTOR:
                equipment = Motor(
                    name=name, equipment_type=equipment_type, voltage_kv=voltage_kv, kva=kva,
                    locked_rotor_multiplier=lr_mult, service_factor=sf, efficiency=eff
                )
            else:
                equipment = Equipment(
                    name=name, equipment_type=equipment_type, voltage_kv=voltage_kv, kva=kva
                )
            
            st.session_state.equipment_list.append(equipment)
            st.session_state.show_add_form = False
            st.success(f"✅ Added {name}")
            st.rerun()

def display_equipment_details(equipment: Equipment, idx: int):
    """Display equipment details with calculations"""
    # Basic info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Type", equipment.equipment_type.value)
    with col2:
        st.metric("Voltage", f"{equipment.voltage_kv} kV")
    with col3:
        i_fl = equipment.calculate_full_load_current()
        st.metric("Full Load Current", f"{i_fl:.2f} A")
    
    # Type-specific details
    if isinstance(equipment, Transformer):
        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric("Impedance", f"{equipment.impedance_percent}%")
        with col5:
            i_sc = equipment.calculate_short_circuit_current()
            st.metric("Isc", f"{i_sc:.0f} A")
        with col6:
            i_inrush = equipment.calculate_inrush_current()
            st.metric("Inrush", f"{i_inrush:.0f} A")
    
    elif isinstance(equipment, Generator):
        col4, col5 = st.columns(2)
        with col4:
            st.metric("X\"d", f"{equipment.xd_subtransient} pu")
        with col5:
            i_subtrans = equipment.calculate_subtransient_current()
            st.metric("I\" (subtransient)", f"{i_subtrans:.0f} A")
    
    elif isinstance(equipment, Cable):
        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric("Length", f"{equipment.length_m} m")
        with col5:
            z = equipment.calculate_impedance()
            st.metric("Impedance", f"{z:.4f} Ω")
        with col6:
            st.metric("Ampacity", f"{equipment.ampacity} A")
    
    # Action buttons
    col_act1, col_act2, col_act3 = st.columns([2, 1, 1])
    with col_act2:
        if st.button(f"⚙️ Settings", key=f"settings_{idx}", use_container_width=True):
            generate_relay_settings(equipment)
    with col_act3:
        if st.button(f"🗑️ Delete", key=f"delete_{idx}", use_container_width=True):
            st.session_state.equipment_list.pop(idx)
            if equipment.name in st.session_state.relay_settings:
                del st.session_state.relay_settings[equipment.name]
            st.rerun()

def generate_relay_settings(equipment: Equipment):
    """Generate recommended relay settings for equipment"""
    recommender = RecommendationEngine()
    
    if isinstance(equipment, Transformer):
        settings = recommender.recommend_transformer_settings(equipment)
    elif isinstance(equipment, Generator):
        settings = recommender.recommend_generator_settings(equipment)
    elif isinstance(equipment, UPS):
        settings = recommender.recommend_ups_settings(equipment)
    elif isinstance(equipment, Cable):
        # Need downstream load current - use 0 for now
        settings = recommender.recommend_cable_settings(equipment, equipment.calculate_full_load_current())
    elif isinstance(equipment, Motor):
        settings = recommender.recommend_motor_settings(equipment)
    else:
        st.warning("No recommendation engine for this equipment type")
        return
    
    st.session_state.relay_settings[equipment.name] = settings
    st.success(f"✅ Generated settings for {equipment.name}")
    st.rerun()

def relay_settings_tab():
    """Tab for relay settings configuration"""
    st.header("⚙️ Relay Protection Settings")
    
    if not st.session_state.relay_settings:
        st.info("💡 Generate relay settings from the Equipment Data tab first.")
        return
    
    for equipment_name, settings in st.session_state.relay_settings.items():
        with st.expander(f"🛡️ {equipment_name}", expanded=True):
            display_relay_settings_form(equipment_name, settings)

def display_relay_settings_form(equipment_name: str, settings: RelaySettings):
    """Display and edit relay settings"""
    
    # Override option
    col1, col2 = st.columns([1, 1])
    with col1:
        adopt_recommended = st.checkbox(
            "✅ Adopt Recommended Settings",
            value=not settings.use_custom,
            key=f"adopt_{equipment_name}"
        )
    with col2:
        use_custom = st.checkbox(
            "✏️ Use Custom Settings",
            value=settings.use_custom,
            key=f"custom_{equipment_name}"
        )
    
    settings.use_custom = use_custom
    
    st.markdown("##### Phase Overcurrent (51P)")
    col_51p1, col_51p2, col_51p3 = st.columns(3)
    
    with col_51p1:
        settings.phase_pickup_current = st.number_input(
            "Pickup Current (A)",
            min_value=0.0,
            value=settings.phase_pickup_current,
            step=1.0,
            disabled=not use_custom,
            key=f"51p_pickup_{equipment_name}"
        )
    
    with col_51p2:
        curve_options = [c.value for c in CurveType]
        curve_idx = curve_options.index(settings.phase_curve_type.value)
        selected_curve = st.selectbox(
            "Curve Type",
            options=curve_options,
            index=curve_idx,
            disabled=not use_custom,
            key=f"51p_curve_{equipment_name}"
        )
        settings.phase_curve_type = CurveType(selected_curve)
    
    with col_51p3:
        settings.phase_tms = st.number_input(
            "TMS",
            min_value=0.05,
            max_value=1.0,
            value=settings.phase_tms,
            step=0.05,
            disabled=not use_custom,
            key=f"51p_tms_{equipment_name}"
        )
    
    # Instantaneous (50P)
    st.markdown("##### Instantaneous Overcurrent (50P)")
    col_50p1, col_50p2 = st.columns([1, 2])
    
    with col_50p1:
        settings.instantaneous_enabled = st.checkbox(
            "Enable 50P",
            value=settings.instantaneous_enabled,
            disabled=not use_custom,
            key=f"50p_enable_{equipment_name}"
        )
    
    with col_50p2:
        if settings.instantaneous_enabled:
            settings.instantaneous_pickup = st.number_input(
                "50P Pickup (A)",
                min_value=0.0,
                value=settings.instantaneous_pickup,
                step=10.0,
                disabled=not use_custom,
                key=f"50p_pickup_{equipment_name}"
            )
    
    # Earth Fault (51N)
    st.markdown("##### Earth Fault (51N)")
    col_51n1, col_51n2, col_51n3 = st.columns(3)
    
    with col_51n1:
        settings.earth_fault_pickup = st.number_input(
            "51N Pickup (A)",
            min_value=0.0,
            value=settings.earth_fault_pickup,
            step=1.0,
            disabled=not use_custom,
            key=f"51n_pickup_{equipment_name}"
        )
    
    with col_51n2:
        ef_curve_options = [c.value for c in CurveType]
        ef_curve_idx = ef_curve_options.index(settings.earth_fault_curve_type.value)
        selected_ef_curve = st.selectbox(
            "51N Curve",
            options=ef_curve_options,
            index=ef_curve_idx,
            disabled=not use_custom,
            key=f"51n_curve_{equipment_name}"
        )
        settings.earth_fault_curve_type = CurveType(selected_ef_curve)
    
    with col_51n3:
        settings.earth_fault_tms = st.number_input(
            "51N TMS",
            min_value=0.05,
            max_value=1.0,
            value=settings.earth_fault_tms,
            step=0.05,
            disabled=not use_custom,
            key=f"51n_tms_{equipment_name}"
        )
    
    # Special protections
    if settings.second_harmonic_blocking or settings.reverse_power_enabled:
        st.markdown("##### Special Protections")
        col_sp1, col_sp2 = st.columns(2)
        
        with col_sp1:
            if settings.second_harmonic_blocking:
                st.checkbox(
                    "2nd Harmonic Blocking",
                    value=True,
                    disabled=True,
                    key=f"2h_{equipment_name}"
                )
        
        with col_sp2:
            if settings.reverse_power_enabled:
                settings.reverse_power_percent = st.number_input(
                    "Reverse Power (%)",
                    min_value=0.0,
                    value=settings.reverse_power_percent,
                    step=0.5,
                    disabled=not use_custom,
                    key=f"rev_pwr_{equipment_name}"
                )
    
    # Advanced mode details
    if st.session_state.advanced_mode:
        with st.expander("🔬 Advanced: IEC Curve Equation"):
            st.latex(r"t = \frac{k \times TMS}{(I/I_{pickup})^{\alpha} - 1}")
            
            constants = IEC_CURVE_CONSTANTS[settings.phase_curve_type]
            st.write(f"**{settings.phase_curve_type.value}**")
            st.write(f"k = {constants['k']}")
            st.write(f"α = {constants['alpha']}")
            st.write(f"TMS = {settings.phase_tms}")

def coordination_results_tab():
    """Tab for coordination validation results"""
    st.header("🔍 Coordination Validation")
    
    if len(st.session_state.relay_settings) < 2:
        st.info("💡 Add at least 2 equipment with relay settings to validate coordination.")
        return
    
    st.markdown("### Coordination Validation Matrix")
    
    # Get list of equipment with settings
    equipment_names = list(st.session_state.relay_settings.keys())
    
    # Fault current input
    col_fault1, col_fault2 = st.columns([2, 1])
    with col_fault1:
        fault_current = st.number_input(
            "Fault Current for CTI Check (A)",
            min_value=0.0,
            value=10000.0,
            step=1000.0,
            help="Fault current at the coordination point"
        )
    
    # Pairwise coordination checks
    validator = CoordinationValidator()
    calc = ProtectionCalculator()
    
    results_data = []
    
    for i in range(len(equipment_names) - 1):
        upstream_name = equipment_names[i]
        downstream_name = equipment_names[i + 1]
        
        upstream_settings = st.session_state.relay_settings[upstream_name]
        downstream_settings = st.session_state.relay_settings[downstream_name]
        
        # Find equipment data if available
        upstream_equipment = next(
            (eq for eq in st.session_state.equipment_list if eq.name == upstream_name),
            None
        )
        
        # Validate coordination
        result = validator.validate_coordination(
            upstream_settings,
            downstream_settings,
            fault_current,
            upstream_equipment
        )
        
        # Display results
        with st.expander(f"📊 {upstream_name} ⬆️ ← {downstream_name} ⬇️", expanded=True):
            
            # Status indicator
            status_color = {
                CoordinationStatus.OK: "green",
                CoordinationStatus.MARGINAL: "orange",
                CoordinationStatus.FAILED: "red"
            }
            
            st.markdown(f"### {result['overall_status'].value}")
            
            # Metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            with col_m1:
                st.metric(
                    "Pickup Selectivity",
                    "✅" if result['pickup_ok'] else "❌"
                )
            
            with col_m2:
                st.metric("CTI", f"{result['cti_value']:.3f}s")
            
            with col_m3:
                t_down = calc.calculate_iec_curve_time(
                    fault_current,
                    downstream_settings.phase_pickup_current,
                    downstream_settings.phase_tms,
                    downstream_settings.phase_curve_type
                )
                st.metric("Downstream Time", f"{t_down:.3f}s")
            
            with col_m4:
                t_up = calc.calculate_iec_curve_time(
                    fault_current,
                    upstream_settings.phase_pickup_current,
                    upstream_settings.phase_tms,
                    upstream_settings.phase_curve_type
                )
                st.metric("Upstream Time", f"{t_up:.3f}s")
            
            # Messages
            st.markdown("#### Validation Details")
            for message in result['messages']:
                st.markdown(f"- {message}")
            
            # Advanced mode calculations
            if st.session_state.advanced_mode:
                st.markdown("---")
                st.markdown("#### 🔬 Advanced: Detailed Calculations")
                
                st.markdown(f"**Downstream Relay ({downstream_name})**")
                st.latex(f"t_{{down}} = \\frac{{k \\times TMS}}{{(I/I_{{pickup}})^{{\\alpha}} - 1}}")
                
                constants_down = IEC_CURVE_CONSTANTS[downstream_settings.phase_curve_type]
                ratio_down = fault_current / downstream_settings.phase_pickup_current
                st.write(f"k = {constants_down['k']}, α = {constants_down['alpha']}, TMS = {downstream_settings.phase_tms}")
                st.write(f"I/I_pickup = {fault_current:.0f}/{downstream_settings.phase_pickup_current:.2f} = {ratio_down:.2f}")
                st.write(f"t_down = {t_down:.3f} seconds")
                
                st.markdown(f"**Upstream Relay ({upstream_name})**")
                constants_up = IEC_CURVE_CONSTANTS[upstream_settings.phase_curve_type]
                ratio_up = fault_current / upstream_settings.phase_pickup_current
                st.write(f"k = {constants_up['k']}, α = {constants_up['alpha']}, TMS = {upstream_settings.phase_tms}")
                st.write(f"I/I_pickup = {fault_current:.0f}/{upstream_settings.phase_pickup_current:.2f} = {ratio_up:.2f}")
                st.write(f"t_up = {t_up:.3f} seconds")
                
                st.markdown("**CTI Calculation**")
                st.latex("CTI = t_{upstream} - t_{downstream}")
                st.write(f"CTI = {t_up:.3f} - {t_down:.3f} = {result['cti_value']:.3f} seconds")
                
                if result['cti_value'] >= IEEE_242_DEFAULTS['recommended_cti']:
                    st.success(f"✅ CTI ≥ {IEEE_242_DEFAULTS['recommended_cti']}s (Recommended)")
                elif result['cti_value'] >= IEEE_242_DEFAULTS['minimum_cti']:
                    st.warning(f"⚠️ CTI ≥ {IEEE_242_DEFAULTS['minimum_cti']}s (Minimum acceptable)")
                else:
                    st.error(f"❌ CTI < {IEEE_242_DEFAULTS['minimum_cti']}s (Not coordinated)")
        
        # Store for summary
        results_data.append({
            "Upstream": upstream_name,
            "Downstream": downstream_name,
            "Status": result['overall_status'].value,
            "CTI (s)": f"{result['cti_value']:.3f}",
            "Pickup OK": "✅" if result['pickup_ok'] else "❌",
            "CTI OK": "✅" if result['cti_ok'] else "❌"
        })
    
    # Summary table
    if results_data:
        st.markdown("---")
        st.markdown("### 📋 Coordination Summary")
        df_results = pd.DataFrame(results_data)
        st.dataframe(df_results, use_container_width=True)

def summary_report_tab():
    """Tab for generating summary report"""
    st.header("📄 Summary Report")
    
    st.markdown(f"### Project: {st.session_state.project_name}")
    st.markdown(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    st.markdown(f"**Standards:** IEEE 242-2001, IEC 60255")
    
    st.markdown("---")
    
    # Equipment Summary
    st.subheader("⚙️ Equipment Summary")
    
    if st.session_state.equipment_list:
        equipment_data = []
        for eq in st.session_state.equipment_list:
            i_fl = eq.calculate_full_load_current()
            equipment_data.append({
                "Name": eq.name,
                "Type": eq.equipment_type.value,
                "Voltage (kV)": eq.voltage_kv,
                "Rating (kVA)": eq.kva if hasattr(eq, 'kva') and eq.kva > 0 else "-",
                "Full Load Current (A)": f"{i_fl:.2f}"
            })
        
        df_equipment = pd.DataFrame(equipment_data)
        st.dataframe(df_equipment, use_container_width=True)
    else:
        st.info("No equipment added yet.")
    
    st.markdown("---")
    
    # Relay Settings Summary
    st.subheader("⚙️ Relay Settings Summary")
    
    if st.session_state.relay_settings:
        settings_data = []
        for name, settings in st.session_state.relay_settings.items():
            settings_data.append({
                "Equipment": name,
                "51P Pickup (A)": f"{settings.phase_pickup_current:.2f}",
                "51P Curve": settings.phase_curve_type.value,
                "51P TMS": settings.phase_tms,
                "50P Enabled": "Yes" if settings.instantaneous_enabled else "No",
                "50P Pickup (A)": f"{settings.instantaneous_pickup:.0f}" if settings.instantaneous_enabled else "-",
                "51N Pickup (A)": f"{settings.earth_fault_pickup:.2f}"
            })
        
        df_settings = pd.DataFrame(settings_data)
        st.dataframe(df_settings, use_container_width=True)
    else:
        st.info("No relay settings generated yet.")
    
    st.markdown("---")
    
    # Export options
    st.subheader("📥 Export Options")
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        if st.button("📊 Export to CSV", use_container_width=True, type="primary"):
            export_to_csv()
    
    with col_exp2:
        if st.button("📋 Copy Summary", use_container_width=True):
            summary_text = generate_summary_text()
            st.code(summary_text, language="text")
            st.success("✅ Summary generated! Copy the text above.")

def export_to_csv():
    """Export all data to CSV files"""
    
    # Equipment data
    if st.session_state.equipment_list:
        equipment_data = []
        for eq in st.session_state.equipment_list:
            data = {
                "Name": eq.name,
                "Type": eq.equipment_type.value,
                "Voltage_kV": eq.voltage_kv,
                "kVA": eq.kva if hasattr(eq, 'kva') else 0,
                "Full_Load_Current_A": eq.calculate_full_load_current()
            }
            
            # Add type-specific data
            if isinstance(eq, Transformer):
                data.update({
                    "Impedance_%": eq.impedance_percent,
                    "Inrush_Mult": eq.inrush_multiplier,
                    "Short_Circuit_A": eq.calculate_short_circuit_current()
                })
            elif isinstance(eq, Generator):
                data.update({
                    "Xd_pu": eq.xd_subtransient,
                    "Subtransient_A": eq.calculate_subtransient_current()
                })
            elif isinstance(eq, Cable):
                data.update({
                    "Length_m": eq.length_m,
                    "Impedance_ohm": eq.calculate_impedance(),
                    "Ampacity_A": eq.ampacity
                })
            
            equipment_data.append(data)
        
        df_equipment = pd.DataFrame(equipment_data)
        
        # Create CSV buffer
        csv_buffer = io.StringIO()
        df_equipment.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="⬇️ Download Equipment Data CSV",
            data=csv_data,
            file_name=f"{st.session_state.project_name}_equipment.csv",
            mime="text/csv"
        )
    
    # Relay settings data
    if st.session_state.relay_settings:
        settings_data = []
        for name, settings in st.session_state.relay_settings.items():
            settings_data.append({
                "Equipment": name,
                "51P_Pickup_A": settings.phase_pickup_current,
                "51P_Curve": settings.phase_curve_type.value,
                "51P_TMS": settings.phase_tms,
                "50P_Enabled": settings.instantaneous_enabled,
                "50P_Pickup_A": settings.instantaneous_pickup if settings.instantaneous_enabled else 0,
                "51N_Pickup_A": settings.earth_fault_pickup,
                "51N_Curve": settings.earth_fault_curve_type.value,
                "51N_TMS": settings.earth_fault_tms,
                "Custom": settings.use_custom
            })
        
        df_settings = pd.DataFrame(settings_data)
        
        csv_buffer2 = io.StringIO()
        df_settings.to_csv(csv_buffer2, index=False)
        csv_data2 = csv_buffer2.getvalue()
        
        st.download_button(
            label="⬇️ Download Relay Settings CSV",
            data=csv_data2,
            file_name=f"{st.session_state.project_name}_relay_settings.csv",
            mime="text/csv"
        )

def generate_summary_text() -> str:
    """Generate text summary of the study"""
    summary = f"""
{'='*70}
PDC ASSISTANT - PROTECTION COORDINATION STUDY
{'='*70}

Project: {st.session_state.project_name}
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
Standards: IEEE 242-2001, IEC 60255

{'='*70}
EQUIPMENT SUMMARY
{'='*70}

"""
    
    for eq in st.session_state.equipment_list:
        i_fl = eq.calculate_full_load_current()
        summary += f"\n{eq.name} ({eq.equipment_type.value})\n"
        summary += f"  Voltage: {eq.voltage_kv} kV\n"
        summary += f"  Rating: {eq.kva} kVA\n"
        summary += f"  Full Load Current: {i_fl:.2f} A\n"
        
        if isinstance(eq, Transformer):
            summary += f"  Impedance: {eq.impedance_percent}%\n"
            summary += f"  Short Circuit: {eq.calculate_short_circuit_current():.0f} A\n"
        elif isinstance(eq, Generator):
            summary += f"  X\"d: {eq.xd_subtransient} pu\n"
            summary += f"  Subtransient: {eq.calculate_subtransient_current():.0f} A\n"
    
    summary += f"\n{'='*70}\n"
    summary += "RELAY SETTINGS\n"
    summary += f"{'='*70}\n\n"
    
    for name, settings in st.session_state.relay_settings.items():
        summary += f"\n{name}:\n"
        summary += f"  51P: {settings.phase_pickup_current:.2f}A, {settings.phase_curve_type.value}, TMS={settings.phase_tms}\n"
        if settings.instantaneous_enabled:
            summary += f"  50P: {settings.instantaneous_pickup:.0f}A\n"
        summary += f"  51N: {settings.earth_fault_pickup:.2f}A, TMS={settings.earth_fault_tms}\n"
    
    summary += f"\n{'='*70}\n"
    
    return summary

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    
    # Page configuration
    st.set_page_config(
        page_title="PDC Assistant",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {padding-top: 0rem;}
        .stMetric {background-color: #f0f2f6; padding: 10px; border-radius: 5px;}
        .stExpander {border: 1px solid #e0e0e0; border-radius: 5px;}
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    init_session_state()
    
    # Sidebar
    sidebar_setup()
    
    # Main header
    st.title("⚡ PDC Assistant")
    st.markdown("**Protection & Device Coordination Assistant** | IEEE 242 & IEC 60255 Compliant")
    st.markdown("---")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "⚙️ Equipment Data",
        "🛡️ Relay Settings",
        "🔍 Coordination Results",
        "📄 Summary Report"
    ])
    
    with tab1:
        equipment_data_tab()
    
    with tab2:
        relay_settings_tab()
    
    with tab3:
        coordination_results_tab()
    
    with tab4:
        summary_report_tab()
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: gray; font-size: 12px;'>
        PDC Assistant v1.0.0 | IEEE 242-2001 & IEC 60255 Compliant<br>
        Senior Power System Protection Engineering Tool
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
