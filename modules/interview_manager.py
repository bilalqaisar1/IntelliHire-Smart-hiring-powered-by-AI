"""
Interview Manager Module

This module handles the AI interview flow, question management, and scoring system.
"""

from typing import Dict, Any, List
import streamlit as st
from modules.agent.orchestrator import AgentOrchestrator
from modules.enhanced_evaluation import generate_correct_answers, evaluate_with_reference_answers, display_detailed_evaluation_results


class InterviewManager:
    """Manages the AI interview process and scoring."""
    
    def __init__(self):
        self.orchestrator = AgentOrchestrator()
    
    def start_interview(self, analysis_result: Dict[str, Any]) -> bool:
        """
        Start the interview process.
        
        Args:
            analysis_result: The analysis result from orchestrator
            
        Returns:
            True if interview started successfully, False otherwise
        """
        try:
            with st.spinner("🧩 Generating interview questions..."):
                questions = self.orchestrator.prepare_interview(analysis_result)
            
            if not questions:
                st.warning("No interview questions could be generated. Please try with a different resume.")
                return False
            
            # Initialize interview session
            st.session_state["questions"] = questions
            st.session_state["answers"] = {}
            st.session_state["current_q"] = 0
            st.session_state["interview_started"] = True
            st.session_state["interview_scores"] = {}
            
            # Store matched skills for reference answer generation
            matched_skills = self.orchestrator.get_matched_skills_from_result(analysis_result)
            st.session_state["matched_skills"] = matched_skills
            
            return True
            
        except Exception as e:
            st.error(f"Failed to start interview: {e}")
            return False
    
    def display_interview_progress(self, analysis_result: Dict[str, Any]):
        """Display interview progress and current question."""
        questions = st.session_state["questions"]
        answers = st.session_state["answers"]
        current_q = st.session_state["current_q"]
        
        # Show compact header during interview
        st.markdown("---")
        st.header("🎤 AI Interview in Progress")
        
        # Progress indicator
        progress = (current_q + 1) / len(questions)
        st.progress(progress)
        st.caption(f"Question {current_q + 1} of {len(questions)}")
        
        # Show current question
        st.markdown(f"### 🧠 Question {current_q + 1}")
        st.info(questions[current_q])
        
        # Answer input
        user_answer = st.text_area(
            "💬 Your Answer:", 
            key=f"ans_{current_q}",
            height=150,
            placeholder="Type your answer here..."
        )
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if current_q > 0:
                if st.button("⬅️ Previous", use_container_width=True, key="prev_question"):
                    # Save current answer
                    answers[current_q] = user_answer.strip()
                    st.session_state["current_q"] -= 1
                    st.rerun()
        
        with col2:
            if current_q + 1 < len(questions):
                if st.button("Next ➡️", use_container_width=True, key="next_question"):
                    # Save current answer and move to next
                    answers[current_q] = user_answer.strip()
                    st.session_state["current_q"] += 1
                    st.rerun()
            else:
                if st.button("🏁 Finish Interview", use_container_width=True, key="finish_interview"):
                    # Save final answer and complete interview
                    answers[current_q] = user_answer.strip()
                    self.complete_interview(questions, answers)
        
        with col3:
            if st.button("❌ Cancel Interview", use_container_width=True, key="cancel_interview"):
                if st.session_state.get("confirm_cancel"):
                    self.cancel_interview()
                else:
                    st.session_state["confirm_cancel"] = True
                    st.rerun()
        
        # Show cancel confirmation
        if st.session_state.get("confirm_cancel"):
            st.warning("⚠️ Are you sure you want to cancel the interview? All progress will be lost.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, Cancel", use_container_width=True, key="confirm_cancel_yes"):
                    self.cancel_interview()
            with col2:
                if st.button("❌ No, Continue", use_container_width=True, key="confirm_cancel_no"):
                    del st.session_state["confirm_cancel"]
                    st.rerun()
    
    def complete_interview(self, questions: List[str], answers: Dict[int, str]):
        """Complete the interview and calculate scores using enhanced evaluation."""
        try:
            with st.spinner("🧾 Evaluating your answers..."):
                # Get matched skills for generating reference answers
                matched_skills = st.session_state.get("matched_skills", [])
                
                # Generate reference answers
                with st.spinner("📚 Generating reference answers..."):
                    correct_answers = generate_correct_answers(questions, matched_skills)
                
                # Evaluate with reference answers
                evaluation_result = evaluate_with_reference_answers(questions, answers, correct_answers)
                
                # Store results
                st.session_state["interview_completed"] = True
                st.session_state["interview_score"] = evaluation_result.get("overall_score", 0.0)
                st.session_state["interview_questions"] = questions
                st.session_state["interview_answers"] = answers
                st.session_state["evaluation_result"] = evaluation_result
                st.session_state["correct_answers"] = correct_answers
                
                # Clear interview state
                if "interview_started" in st.session_state:
                    del st.session_state["interview_started"]
                if "current_q" in st.session_state:
                    del st.session_state["current_q"]
                if "confirm_cancel" in st.session_state:
                    del st.session_state["confirm_cancel"]
                
                st.rerun()
                
        except Exception as e:
            st.error(f"Failed to complete interview: {e}")
            # Fallback to original scoring
            try:
                interview_score = self.orchestrator.score_interview(questions, answers)
                st.session_state["interview_completed"] = True
                st.session_state["interview_score"] = interview_score
                st.session_state["interview_questions"] = questions
                st.session_state["interview_answers"] = answers
                st.rerun()
            except Exception as e2:
                st.error(f"Fallback scoring also failed: {e2}")
    
    def cancel_interview(self):
        """Cancel the interview and return to eligibility screen."""
        # Clear interview state
        keys_to_clear = [
            'interview_started', 'questions', 'answers', 'current_q',
            'confirm_cancel', 'interview_scores'
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        st.rerun()
    
    def display_interview_results(self, analysis_result: Dict[str, Any]):
        """Display final interview results and scores with enhanced evaluation."""
        st.markdown("---")
        st.header("🏆 Interview Results")
        
        # Get scores
        interview_score = st.session_state.get("interview_score", 0)
        match_score = analysis_result.get('overall_score', 0)
        
        # Display scores prominently
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "🎯 Resume Match Score", 
                f"{match_score:.1f}%",
                help="How well your resume matches the job requirements"
            )
        
        with col2:
            st.metric(
                "🧠 Interview Score", 
                f"{interview_score:.1f}%",
                help="Your performance in the AI interview"
            )
        
        # Overall assessment
        st.markdown("---")
        st.subheader("📊 Overall Assessment")
        
        # Calculate overall score (weighted average)
        overall_score = (match_score * 0.4) + (interview_score * 0.6)
        
        st.metric("🏆 Overall Score", f"{overall_score:.1f}%")
        st.progress(overall_score / 100)
        
        # Performance feedback
        if overall_score >= 80:
            st.success("🌟 Excellent! You performed very well in both resume matching and interview.")
        elif overall_score >= 70:
            st.success("✅ Good performance! You meet the requirements well.")
        elif overall_score >= 60:
            st.warning("⚠️ Fair performance. Consider improving in weaker areas.")
        else:
            st.error("❌ Below expectations. Significant improvement needed.")
        
        # Enhanced detailed evaluation if available
        evaluation_result = st.session_state.get("evaluation_result")
        questions = st.session_state.get("interview_questions", [])
        
        if evaluation_result and questions:
            st.markdown("---")
            display_detailed_evaluation_results(evaluation_result, questions)
        else:
            # Fallback to simple Q&A display
            st.markdown("---")
            st.subheader("📋 Interview Questions & Answers")
            
            questions = st.session_state.get("interview_questions", [])
            answers = st.session_state.get("interview_answers", {})
            
            if questions and answers:
                for i, question in enumerate(questions):
                    answer = answers.get(i, "No answer provided")
                    with st.expander(f"Question {i+1}: {question[:50]}..."):
                        st.markdown(f"**Question:** {question}")
                        st.markdown(f"**Your Answer:** {answer}")
        
        # Action buttons
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Try Different Resume", use_container_width=True, key="try_different_results"):
                self.reset_all_sessions()
                st.rerun()
        
        with col2:
            if st.button("📊 View Detailed Analysis", use_container_width=True, key="view_detailed_results"):
                st.session_state['show_detailed_analysis'] = True
                st.rerun()
        
        with col3:
            if st.button("🏠 Start Over", use_container_width=True, key="start_over_results"):
                self.reset_all_sessions()
                st.rerun()
    
    def reset_all_sessions(self):
        """Reset all session state for a fresh start."""
        keys_to_clear = [
            'analysis_result', 'interview_started', 'questions', 
            'answers', 'current_q', 'show_compact_interview_header',
            'eligibility_status', 'interview_completed', 'interview_score',
            'interview_questions', 'interview_answers', 'interview_scores',
            'confirm_cancel', 'show_detailed_analysis'
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
