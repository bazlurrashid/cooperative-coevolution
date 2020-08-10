package ccfsrfg2;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.io.File;

import javax.swing.Action;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.ButtonGroup;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JPopupMenu;
import javax.swing.JRadioButton;
import javax.swing.JScrollPane;
import javax.swing.JSplitPane;
import javax.swing.JTextArea;
import javax.swing.KeyStroke;
import javax.swing.filechooser.FileNameExtensionFilter;
import javax.swing.text.JTextComponent;
import javax.swing.text.TextAction;

public class CCEAFS extends Thread{
	
	private JFrame jFrame;
	private JSplitPane splitPaneH;
	private JPanel	jPanel1;
	private JPanel  jPanel2;
	private JPanel jPanel;
	private JButton jOpenButton;
	private JButton jTestButton;
	private JButton jStartButton;
	private JButton jResetButton;
	private JRadioButton jTest;
	private JRadioButton jCross;
	private JRadioButton jSplit;
	private ButtonGroup buttonGroup1;
	private JRadioButton jNaiveBayes;
	private JRadioButton jSVM;
	private JRadioButton jLibSVM;
	private JRadioButton jKNN;
	private JRadioButton jJ48;
	private JRadioButton jRandomForest;
	private JRadioButton jLogisticRegression;
	private ButtonGroup buttonGroup2;
	private JTextArea jTextArea;
	private JScrollPane jScrollPane;
	private JLabel jLabel;
	private JPopupMenu jPopupMenu;
    
	private File selectedFile = null;
	private String absolutePath;
	private String fileNameWithExtension;
	private String fileName;
	
	private File selectedTestFile = null;
	private String absoluteTestPath;
	private String testFileNameWithExtension;
	private String testFileName;
	
	private static String newLine;
	private String classificationMode;
	private String classificationModel;
	private Boolean startButtonClicked;
	private Boolean resetButtonClicked;
	
	public CCEAFS() {
		
		startButtonClicked = false;
		newLine = "\n";
		jFrame = new JFrame();
		jFrame.setLayout(new FlowLayout());
		jFrame.setVisible(true);
		jPanel = new JPanel(new BorderLayout());
        jFrame.setContentPane(jPanel);
        jPanel1 = new JPanel();
        jPanel1.setLayout(new BoxLayout(jPanel1, BoxLayout.Y_AXIS));
		jPanel2 = new JPanel(new BorderLayout());
		splitPaneH = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT);
		jPanel.add(splitPaneH, BorderLayout.CENTER);
		splitPaneH.setLeftComponent(jPanel1);
		splitPaneH.setRightComponent(jPanel2);

        jOpenButton = new JButton("Open dataset..");
        jTestButton = new JButton("Open test dataset..");
        jStartButton = new JButton("Start");
        jResetButton = new JButton("Reset");
		jTest = new JRadioButton("Supplied test set");
        jCross = new JRadioButton("Cross-validation (CV)");
        jSplit = new JRadioButton("Percentage split");
        jStartButton.setEnabled(false);
        jResetButton.setEnabled(false);
        jTest.setEnabled(false);
        jCross.setEnabled(false);
        jSplit.setEnabled(false);
        buttonGroup1 = new ButtonGroup();
        
		jNaiveBayes = new JRadioButton("Naive Bayes (NB)");
        jSVM = new JRadioButton("Support Vector Machine (SVM)");
        jLibSVM = new JRadioButton("Lib Support Vector Machine (LibSVM)");
        jKNN = new JRadioButton("K-Nearest Neighbour (KNN)");
        jJ48 = new JRadioButton("J48");
        jRandomForest = new JRadioButton("Random Forest (RF)");
        jLogisticRegression = new JRadioButton("Logistic Regression (LR)");
        jNaiveBayes.setEnabled(false);
        jSVM.setEnabled(false);
        jLibSVM.setEnabled(false);
        jKNN.setEnabled(false);
        jJ48.setEnabled(false);
        jRandomForest.setEnabled(false);
        jLogisticRegression.setEnabled(false);
        
        buttonGroup2 = new ButtonGroup();
        jOpenButton.setMnemonic(KeyEvent.VK_O);
        jOpenButton.setRolloverEnabled(false);
        jTestButton.setMnemonic(KeyEvent.VK_O);
        jTestButton.setEnabled(false);
        jTest.setMnemonic(KeyEvent.VK_T);
        jCross.setMnemonic(KeyEvent.VK_C);
        jSplit.setMnemonic(KeyEvent.VK_S);
        
        buttonGroup1.add(jTest);					buttonGroup1.add(jTestButton);
        buttonGroup1.add(jCross);					buttonGroup1.add(jSplit);
        
        buttonGroup2.add(jNaiveBayes);				buttonGroup2.add(jSVM);
        buttonGroup2.add(jLibSVM);					buttonGroup2.add(jKNN);
        buttonGroup2.add(jJ48);						buttonGroup2.add(jRandomForest);
        buttonGroup2.add(jLogisticRegression);
        
        jPanel1.setPreferredSize(new Dimension(300, 400));
        jLabel = new JLabel("Click here to select a dataset (arff format) for training.");
        jPanel1.add(jLabel);				jPanel1.add(Box.createRigidArea(new Dimension(0, 10)));
        jPanel1.add(jOpenButton);			jPanel1.add(Box.createRigidArea(new Dimension(0, 40)));
        jLabel = new JLabel("Choose one of the classification mode:");
        jPanel1.add(jLabel);				jPanel1.add(Box.createRigidArea(new Dimension(0, 10)));
        jPanel1.add(jTest);					jPanel1.add(Box.createRigidArea(new Dimension(0, 10)));
        jPanel1.add(jTestButton);			jPanel1.add(Box.createRigidArea(new Dimension(0, 10)));
        jPanel1.add(jCross);				jPanel1.add(Box.createRigidArea(new Dimension(0, 10)));
        jPanel1.add(jSplit);				jPanel1.add(Box.createRigidArea(new Dimension(0, 40)));
        jLabel = new JLabel("Choose one of the classification model:");
        jPanel1.add(jLabel);				jPanel1.add(Box.createRigidArea(new Dimension(0, 10)));
        jPanel1.add(jNaiveBayes);			jPanel1.add(Box.createRigidArea(new Dimension(0, 10)));
        jPanel1.add(jSVM);					jPanel1.add(Box.createRigidArea(new Dimension(0, 10)));
        jPanel1.add(jLibSVM);				jPanel1.add(Box.createRigidArea(new Dimension(0, 10)));
        jPanel1.add(jKNN);					jPanel1.add(Box.createRigidArea(new Dimension(0, 10)));
        jPanel1.add(jJ48);					jPanel1.add(Box.createRigidArea(new Dimension(0, 10)));
        jPanel1.add(jRandomForest);			jPanel1.add(Box.createRigidArea(new Dimension(0, 10)));
        jPanel1.add(jLogisticRegression);	jPanel1.add(Box.createRigidArea(new Dimension(0, 40)));
        jLabel = new JLabel("To execute the program click on Start button:");
        jPanel1.add(jLabel);				jPanel1.add(Box.createRigidArea(new Dimension(0, 10)));
        jPanel1.add(jStartButton);
        jLabel = new JLabel("To reset the program click on Reset button:");
        jPanel1.add(jLabel);				jPanel1.add(Box.createRigidArea(new Dimension(0, 10)));
        jPanel1.add(jResetButton);
        
        jPanel1.setVisible(true);	jPanel1.revalidate();	jPanel1.repaint();
        
		jTextArea = new JTextArea(40, 100);
		jTextArea.setBackground(Color.DARK_GRAY);
		jTextArea.setForeground(Color.WHITE);
		jTextArea.setOpaque(true);
		jTextArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 14));
		jTextArea.setCaretPosition(jTextArea.getDocument().getLength());
		jTextArea.setVisible(true);
      
		jScrollPane = new JScrollPane(jTextArea);
		jScrollPane.setViewportView(jTextArea);
		jScrollPane.getViewport().setOpaque(false);
		jPanel2.add(jScrollPane);
		jPanel2.setVisible(true);
        jTextArea.append("A cooperative co-evolutionary framework for feature selection!" + newLine + newLine);
        jFrame.setSize(1000, 800);
        jFrame.setLocationRelativeTo(null);
        jFrame.setTitle("CCFSRFG-2: Cooperative Co-evolutionary Algorithm for Feature Selection");
        jFrame.setVisible(true);
        jFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        jFrame.pack();
        
        jOpenButton.addActionListener(new ActionListener() { 
            public void actionPerformed(ActionEvent e) { 
        		JFileChooser jFileChooser = new JFileChooser();
                jFileChooser.setCurrentDirectory(new File("C:\\Users\\Bazlur\\Desktop\\feature-selection"));
                jFileChooser.setFileFilter(new FileNameExtensionFilter("Arff Files", "arff"));
                jFileChooser.removeChoosableFileFilter(jFileChooser.getAcceptAllFileFilter());

                int result = jFileChooser.showOpenDialog(new JFrame());

                if (result == JFileChooser.APPROVE_OPTION) {
                    selectedFile = jFileChooser.getSelectedFile();
                    absolutePath = selectedFile.getAbsolutePath();
                    fileNameWithExtension = selectedFile.getName();
            		int pos = fileNameWithExtension.lastIndexOf(".");
            		if (pos > 0 && pos < (fileNameWithExtension.length() - 1)) { 
            		    fileName = fileNameWithExtension.substring(0, pos);
            		}
            		jTextArea.append("Train dataset: " + fileName + newLine);

                    jTest.setEnabled(true);
                    jCross.setEnabled(true);
                    jSplit.setEnabled(false);
                }
                if(result == JFileChooser.CANCEL_OPTION ) {
                	JOptionPane.showMessageDialog(jFrame, "No train dataset selected.");
                	jTextArea.append("No train dataset selected." + newLine);
                }
                if(result == JFileChooser.ERROR_OPTION) {
                	JOptionPane.showMessageDialog(jFrame, "Error!!!");
                }
            }	// end of actionPerformed 
        }); 	// end of jOpenButton.addActionListener
        
        ClassificationOptionActionListener actionListener = new ClassificationOptionActionListener();
        jTest.addActionListener(actionListener);
        jCross.addActionListener(actionListener);
        jSplit.addActionListener(actionListener);
        
        jTestButton.addActionListener(new ActionListener() { 
            public void actionPerformed(ActionEvent e) { 
        		JFileChooser jFileChooser = new JFileChooser();
                jFileChooser.setCurrentDirectory(new File("D:\\feature-selection"));
                jFileChooser.setFileFilter(new FileNameExtensionFilter("Arff Files", "arff"));
                jFileChooser.removeChoosableFileFilter(jFileChooser.getAcceptAllFileFilter());

                int result = jFileChooser.showOpenDialog(new JFrame());

                if (result == JFileChooser.APPROVE_OPTION) {
                    selectedTestFile = jFileChooser.getSelectedFile();
                    absoluteTestPath = selectedTestFile.getAbsolutePath();
                    testFileNameWithExtension = selectedTestFile.getName();
            		int pos = testFileNameWithExtension.lastIndexOf(".");
            		if (pos > 0 && pos < (testFileNameWithExtension.length() - 1)) { 
            		    testFileName = testFileNameWithExtension.substring(0, pos);
            		}
            		jTextArea.append("Test dataset: " + testFileName + newLine);
                }
                if(result == JFileChooser.CANCEL_OPTION ) {
                	JOptionPane.showMessageDialog(jFrame, "No test dataset selected.");
                	jTextArea.append("No test dataset selected." + newLine);
                }
                if(result == JFileChooser.ERROR_OPTION) {
                	JOptionPane.showMessageDialog(jFrame, "Error!!!");
                }

            }	// end of actionPerformed
        });	// end of jTestButton.addActionListener
        
        ClassificationModelActionListener actionListener2 = new ClassificationModelActionListener();
        jNaiveBayes.addActionListener(actionListener2);		jSVM.addActionListener(actionListener2);
        jLibSVM.addActionListener(actionListener2);			jKNN.addActionListener(actionListener2);
        jJ48.addActionListener(actionListener2);			jRandomForest.addActionListener(actionListener2);
        jLogisticRegression.addActionListener(actionListener2);
        
        jStartButton.addActionListener(new ActionListener() { 
            public void actionPerformed(ActionEvent e) { 
            	JButton button = (JButton) e.getSource();
    	        if (button == jStartButton) {
    	        	startButtonClicked = true;
    	        	jTextArea.append("\nCCFSRFG-2 starts executing..." + "\n");
    	        	jStartButton.setEnabled(false);
    	        }
            } 
        });	// end of jStartButton.addActionListener
        
        jResetButton.addActionListener(new ActionListener() { 
            public void actionPerformed(ActionEvent e) { 
            	JButton button = (JButton) e.getSource();
    	        if (button == jResetButton) {
    	        	resetButtonClicked = true;
    	        	jTextArea.append("\nCCFSRFG-2 resets to the beginning..." + "\n");
    	        	jResetButton.setEnabled(false);
    	        }
            } 
        });	// end of jStartButton.addActionListener
	}
	
	public String getTrainAbsolutePath() {
		return absolutePath;
	}
	
	public String getTrainFileName() {
		return fileName;
	}
	
	public String getTrainFileNameWithExtension() {
		return fileNameWithExtension;
	}
	
	public String getTestAbsolutePath() {
		return absoluteTestPath;
	}
	
	public String getTestFileName() {
		return testFileName;
	}
	
	public String getTestFileNameWithExtension() {
		return testFileNameWithExtension;
	}
	
	public String getClassificationMode() {
		return classificationMode;
	}
	
	public String getClassificationModel() {
		return classificationModel;
	}
	
	public Boolean getStartButtonClickedStatus() {
		System.out.print("");
		return startButtonClicked;
	}
	
	public Boolean getResetButtonClickedStatus() {
		System.out.print("");
		return resetButtonClicked;
	}
	
	public JTextArea getJTextArea() {
		return jTextArea;
	}
	
	static class SelectAll extends TextAction
    {
        public SelectAll()
        {
            super("Select All");
            putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control S"));
        }

        public void actionPerformed(ActionEvent e)
        {
            JTextComponent component = getFocusedComponent();
            component.selectAll();
            component.requestFocusInWindow();
        }
    }
	
	class ClassificationOptionActionListener implements ActionListener {
	    @Override
		public void actionPerformed(ActionEvent e) {
	    	JRadioButton button = (JRadioButton) e.getSource();
	        if (button == jTest) {
	        	classificationMode = "Train-Test";
	        	jTestButton.setEnabled(true);
	        }
	        else if (button == jCross) {
	        	classificationMode = "CV";
	        	jTestButton.setEnabled(false);
	        }
	        else if (button == jSplit) {
	        	classificationMode = "Percentage-Split";
	        	jTestButton.setEnabled(false);
	        }
	        
	        jNaiveBayes.setEnabled(true);
	        jSVM.setEnabled(true);
	        jLibSVM.setEnabled(true);
	        jKNN.setEnabled(true);
	        jJ48.setEnabled(true);
	        jRandomForest.setEnabled(true);
	        jLogisticRegression.setEnabled(true);
	        
	    }	// end of actionPerformed
	}	// end of ClassificationOptionActionListener
	
	class ClassificationModelActionListener implements ActionListener {
	    @Override
		public void actionPerformed(ActionEvent e) {
	    	JRadioButton button = (JRadioButton) e.getSource();
	        if (button == jNaiveBayes) {
	        	classificationModel = "NB";
	        }
	        else if (button == jSVM) {
	        	classificationModel = "SVM";
	        }
	        else if (button == jLibSVM) {
	        	classificationModel = "LibSVM";
	        }
	        else if (button == jKNN) {
	        	classificationModel = "KNN";
	        }
	        else if (button == jJ48) {
	        	classificationModel = "J48";
	        }
	        else if (button == jRandomForest) {
	        	classificationModel = "RF";
	        }
	        else if (button == jLogisticRegression) {
	        	classificationModel = "LR";
	        }
	        jStartButton.setEnabled(true);
	    }	// end of actionPerformed
	}	// end of ClassificationModelActionListener
	
}


