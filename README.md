# AR-based-Navigation-for-Stereotactic-Brain-Biopsy
Official repository for the AR-based navigation system in stereotactic brain biopsy. Features a multi-objective optimization model for path planning and hybrid registration. 
The system consists of two main components:
1. Path Planning Module (Python): Implements a constrained multi-objective optimization (MOO) model to calculate optimal biopsy trajectories based on clinical safety criteria.
2.AR Guidance Module (Unity & Vuforia): Handles real-time visualization and intraoperative alignment.
Vuforia Engine “https://developer.vuforia.com/home”: Utilized for QR code recognition as Image Targets.
Hybrid Registration: Managed by the hybridregistration.cs script within the Unity environment.
