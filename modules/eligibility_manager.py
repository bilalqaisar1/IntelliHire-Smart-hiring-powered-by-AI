"""
Eligibility Manager Module

This module handles eligibility checks, score validation, and user flow management
for the resume matching system.
"""

from typing import Dict, Any, Tuple
import streamlit as st


class EligibilityManager:
    """Manages eligibility checks and user flow based on match scores."""
    
    ELIGIBILITY_THRESHOLD = 60.0
    
    def __init__(self):
        self.threshold = self.ELIGIBILITY_THRESHOLD
    
    def check_eligibility(self, match_score: float) -> Tuple[bool, str]:
        """
        Check if candidate is eligible based on match score.
        
        Args:
            match_score: The overall match score (0-100)
            
        Returns:
            Tuple of (is_eligible, message)
        """
        if match_score >= self.threshold:
            return True, f"✅ Congratulations! You are eligible for this position with a match score of {match_score:.1f}%"
        else:
            return False, f"❌ Sorry, you are not eligible for this position. Your match score is {match_score:.1f}%, which is below the required {self.threshold:.1f}% threshold."
    
    def get_eligibility_status(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get comprehensive eligibility status from analysis result.
        
        Args:
            analysis_result: The analysis result from orchestrator
            
        Returns:
            Dictionary containing eligibility information
        """
        match_score = analysis_result.get('overall_score', 0.0)
        is_eligible, message = self.check_eligibility(match_score)
        
        return {
            'is_eligible': is_eligible,
            'match_score': match_score,
            'threshold': self.threshold,
            'message': message,
            'missing_skills': analysis_result.get('missing_skills', []),
            'matched_skills': analysis_result.get('matched_skills', [])
        }
    
    def reset_session_for_new_application(self):
        """Reset session state to allow new application."""
        keys_to_clear = [
            'analysis_result', 'interview_started', 'questions', 
            'answers', 'current_q', 'show_compact_interview_header',
            'eligibility_status', 'rejection_message_spoken'
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
    
    def display_eligibility_result(self, eligibility_status: Dict[str, Any]) -> bool:
        """
        Display eligibility result and handle user interaction.
        
        Args:
            eligibility_status: The eligibility status dictionary
            
        Returns:
            True if user wants to proceed, False if they want to restart
        """
        st.markdown("---")
        st.header("📊 Eligibility Assessment")
        
        # Display match score prominently
        match_score = eligibility_status['match_score']
        st.metric("🎯 Overall Match Score", f"{match_score:.1f}%")
        st.progress(match_score / 100)
        
        # Display eligibility message
        is_eligible = eligibility_status['is_eligible']
        if is_eligible:
            st.success(eligibility_status['message'])
        else:
            st.error(eligibility_status['message'])
            # Speak the rejection message only once
            if not st.session_state.get('rejection_message_spoken', False):
                try:
                    from final_app import speak
                    speak(eligibility_status['message'])
                    st.session_state['rejection_message_spoken'] = True
                except Exception as e:
                    st.warning(f"Could not play voice message: {e}")
        
        # Missing skills are now hidden as per user request
        
        # Display matched skills
        matched_skills = eligibility_status['matched_skills']
        if matched_skills:
            st.subheader("✅ Matched Skills")
            st.write(", ".join(matched_skills))
        
        # Handle user action based on eligibility
        if is_eligible:
            # Show start interview button for eligible candidates
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🧠 Start AI Interview", use_container_width=True, key="start_interview_eligible"):
                    return True
            with col2:
                if st.button("🔄 Try Different Resume", use_container_width=True, key="try_different_eligible"):
                    self.reset_session_for_new_application()
                    st.rerun()
        else:
            # Show restart option for ineligible candidates
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🔄 Try Different Resume", use_container_width=True, key="try_different_ineligible"):
                    self.reset_session_for_new_application()
                    st.rerun()
            with col2:
                if st.button("📋 View Detailed Analysis", use_container_width=True, key="view_detailed_analysis"):
                    st.session_state['show_detailed_analysis'] = True
                    st.rerun()
        
        return False
    
    def display_detailed_analysis(self, analysis_result: Dict[str, Any]):
        """Display detailed analysis for ineligible candidates."""
        st.markdown("---")
        st.header("📋 Detailed Analysis")
        
        # Skills breakdown
        st.subheader("Skills Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Required Skills:**")
            required_skills = analysis_result.get('required_skills', [])
            if required_skills:
                for skill in required_skills:
                    st.markdown(f"- {skill}")
            else:
                st.write("No required skills found")
        
        with col2:
            st.markdown("**Your Skills:**")
            user_skills = analysis_result.get('user_skills', [])
            if user_skills:
                for skill in user_skills:
                    st.markdown(f"- {skill}")
            else:
                st.write("No skills found")
        
        # Experience comparison
        st.subheader("Experience Comparison")
        req_exp = analysis_result.get('required_experience')
        usr_exp = analysis_result.get('user_experience')
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Required Experience", f"{req_exp} years" if req_exp else "Not specified")
        with col2:
            st.metric("Your Experience", f"{usr_exp} years" if usr_exp else "Not specified")
        
        # Action buttons
        st.markdown("---")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔄 Try Different Resume", use_container_width=True, key="try_different_detailed"):
                self.reset_session_for_new_application()
                st.rerun()
        with col2:
            if st.button("← Back to Results", use_container_width=True, key="back_to_results"):
                if 'show_detailed_analysis' in st.session_state:
                    del st.session_state['show_detailed_analysis']
                st.rerun()
