#ifndef CLAD_BASE_FORWARD_MODE_VISITOR_H
#define CLAD_BASE_FORWARD_MODE_VISITOR_H

#include "Compatibility.h"
#include "VisitorBase.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Sema/Sema.h"

#include "llvm/ADT/SmallVector.h"

#include "clad/Differentiator/DerivativeBuilder.h"

#include <array>
#include <stack>
#include <unordered_map>

namespace clad {
/// A visitor for processing the function code in forward mode.
/// Used to compute derivatives by clad::differentiate.
class BaseForwardModeVisitor
    : public clang::ConstStmtVisitor<BaseForwardModeVisitor, StmtDiff>,
      public VisitorBase {
  unsigned m_IndependentVarIndex = ~0;

protected:
  const clang::ValueDecl* m_IndependentVar = nullptr;

public:
  BaseForwardModeVisitor(DerivativeBuilder& builder,
                         const DiffRequest& request);
  ~BaseForwardModeVisitor() override;

  ///\brief Produces the first derivative of a given function.
  ///
  ///\returns The differentiated and potentially created enclosing
  /// context.
  ///
  DerivativeAndOverload Derive() override;

  StmtDiff Visit(const clang::Stmt* S) {
    m_CurVisitedStmt = S;
#ifndef NDEBUG
    // Enable testing of the pretty printing of the state when clad crashes.
    if (const char* Env = std::getenv("CLAD_FORCE_CRASH"))
      std::terminate();
#endif // NDEBUG
    return clang::ConstStmtVisitor<BaseForwardModeVisitor, StmtDiff>::Visit(S);
  }

  virtual void ExecuteInsidePushforwardFunctionBlock() {}

  virtual StmtDiff
  VisitArraySubscriptExpr(const clang::ArraySubscriptExpr* ASE);
  StmtDiff VisitBinaryOperator(const clang::BinaryOperator* BinOp);
  StmtDiff VisitCallExpr(const clang::CallExpr* CE);
  StmtDiff VisitCompoundStmt(const clang::CompoundStmt* CS);
  StmtDiff VisitConditionalOperator(const clang::ConditionalOperator* CO);
  StmtDiff VisitCXXConstCastExpr(const clang::CXXConstCastExpr* CCE);
  StmtDiff VisitCXXBoolLiteralExpr(const clang::CXXBoolLiteralExpr* BL);
  StmtDiff VisitCharacterLiteral(const clang::CharacterLiteral* CL);
  StmtDiff VisitStringLiteral(const clang::StringLiteral* SL);
  StmtDiff VisitCXXDefaultArgExpr(const clang::CXXDefaultArgExpr* DE);
  StmtDiff VisitDeclRefExpr(const clang::DeclRefExpr* DRE);
  StmtDiff VisitDeclStmt(const clang::DeclStmt* DS);
  virtual StmtDiff VisitFloatingLiteral(const clang::FloatingLiteral* FL);
  StmtDiff VisitForStmt(const clang::ForStmt* FS);
  StmtDiff VisitIfStmt(const clang::IfStmt* If);
  StmtDiff VisitImplicitCastExpr(const clang::ImplicitCastExpr* ICE);
  StmtDiff VisitInitListExpr(const clang::InitListExpr* ILE);
  virtual StmtDiff VisitIntegerLiteral(const clang::IntegerLiteral* IL);
  StmtDiff VisitMemberExpr(const clang::MemberExpr* ME);
  StmtDiff VisitParenExpr(const clang::ParenExpr* PE);
  virtual StmtDiff VisitReturnStmt(const clang::ReturnStmt* RS);
  StmtDiff VisitStmt(const clang::Stmt* S);
  StmtDiff VisitUnaryOperator(const clang::UnaryOperator* UnOp);
  // Decl is not Stmt, so it cannot be visited directly.
  virtual DeclDiff<clang::VarDecl>
  DifferentiateVarDecl(const clang::VarDecl* VD);
  virtual DeclDiff<clang::VarDecl>
  DifferentiateVarDecl(const clang::VarDecl* VD, bool ignoreInit);
  /// Shorthand for warning on differentiation of unsupported operators
  void unsupportedOpWarn(clang::SourceLocation loc,
                         llvm::ArrayRef<llvm::StringRef> args = {}) {
    diag(clang::DiagnosticsEngine::Warning, loc,
         "attempt to differentiate unsupported operator,  derivative \
                         set to 0",
         args);
  }
  StmtDiff VisitCXXForRangeStmt(const clang::CXXForRangeStmt* FRS);
  StmtDiff VisitWhileStmt(const clang::WhileStmt* WS);
  StmtDiff VisitDoStmt(const clang::DoStmt* DS);
  StmtDiff VisitContinueStmt(const clang::ContinueStmt* ContStmt);

  StmtDiff VisitSwitchStmt(const clang::SwitchStmt* SS);
  StmtDiff VisitBreakStmt(const clang::BreakStmt* BS);
  StmtDiff VisitCXXConstructExpr(const clang::CXXConstructExpr* CE);
  StmtDiff VisitExprWithCleanups(const clang::ExprWithCleanups* EWC);
  StmtDiff
  VisitMaterializeTemporaryExpr(const clang::MaterializeTemporaryExpr* MTE);
  StmtDiff
  VisitCXXTemporaryObjectExpr(const clang::CXXTemporaryObjectExpr* TOE);
  StmtDiff VisitCXXThisExpr(const clang::CXXThisExpr* CTE);
  StmtDiff VisitCXXNewExpr(const clang::CXXNewExpr* CNE);
  StmtDiff VisitCXXDeleteExpr(const clang::CXXDeleteExpr* CDE);
  StmtDiff
  VisitCXXScalarValueInitExpr(const clang::CXXScalarValueInitExpr* SVIE);
  StmtDiff VisitCXXStaticCastExpr(const clang::CXXStaticCastExpr* CSE);
  StmtDiff VisitCXXFunctionalCastExpr(const clang::CXXFunctionalCastExpr* FCE);
  StmtDiff VisitCXXBindTemporaryExpr(const clang::CXXBindTemporaryExpr* BTE);
  StmtDiff VisitCXXNullPtrLiteralExpr(const clang::CXXNullPtrLiteralExpr* NPL);
  StmtDiff
  VisitUnaryExprOrTypeTraitExpr(const clang::UnaryExprOrTypeTraitExpr* UE);
  StmtDiff VisitPseudoObjectExpr(const clang::PseudoObjectExpr* POE);
  StmtDiff VisitSubstNonTypeTemplateParmExpr(
      const clang::SubstNonTypeTemplateParmExpr* NTTP);
  StmtDiff VisitImplicitValueInitExpr(const clang::ImplicitValueInitExpr* IVIE);
  StmtDiff VisitCStyleCastExpr(const clang::CStyleCastExpr* CSCE);
  StmtDiff VisitNullStmt(const clang::NullStmt* NS) { return StmtDiff{}; };
  StmtDiff
  VisitCXXStdInitializerListExpr(const clang::CXXStdInitializerListExpr* ILE);
  static DeclDiff<clang::StaticAssertDecl>
  DifferentiateStaticAssertDecl(const clang::StaticAssertDecl* SAD);

  virtual std::string GetPushForwardFunctionSuffix();
  virtual DiffMode GetPushForwardMode();

protected:
  /// Helper function for differentiating the switch statement body.
  ///
  /// It manages scopes and blocks for the switch case labels, checks if
  /// compound statement to be differentiated is supported and returns the
  /// active switch case label after processing the given `stmt` argument.
  ///
  /// Scope and and block for the last switch case label have to be managed
  /// manually outside the function because this function have no way of
  /// knowing when all the statements belonging to last switch case label have
  /// been processed.
  ///
  /// \param[in] stmt Current statement to derive
  /// \param[in] activeSC Current active switch case label
  /// \return active switch case label after processing `stmt`
  clang::SwitchCase* DeriveSwitchStmtBodyHelper(const clang::Stmt* stmt,
                                                clang::SwitchCase* activeSC);

  /// Tries to build custom derivative constructor pushforward call for the
  /// given CXXConstructExpr.
  ///
  /// \return A call expression if a suitable custom derivative is found;
  /// Otherwise returns nullptr.
  clang::Expr* BuildCustomDerivativeConstructorPFCall(
      const clang::CXXConstructExpr* CE,
      llvm::SmallVectorImpl<clang::Expr*>& clonedArgs,
      llvm::SmallVectorImpl<clang::Expr*>& derivedArgs);

private:
  /// Prepares the derivative function parameters.
  void
  SetupDerivativeParameters(llvm::SmallVectorImpl<clang::ParmVarDecl*>& params);

  /// Generate a seed initializing each independent argument with 1 and 0
  /// otherwise:
  /// double f_darg0(double x, double y) {
  ///   double _d_x = 1;
  ///   double _d_y = 0;
  void GenerateSeeds(const clang::FunctionDecl* dFD);
};
} // end namespace clad

#endif // CLAD_FORWARD_MODE_VISITOR_H
